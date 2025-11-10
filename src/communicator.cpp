#include "communicator.hpp"
#include "icontroller.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

// ---------- helpers só visíveis neste TU ----------
namespace {

// leitura de linha simples (tolerante a CRLF)
bool read_line(int fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
        ssize_t n = ::recv(fd, &c, 1, 0);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (c == '\n') break;
        if (c == '\r') continue;
        out.push_back(c);
        if (out.size() > 2'000'000) return false;
    }
    return true;
}

bool write_line(int fd, const std::string& s) {
    std::string msg = s + "\n";
    const char* p = msg.c_str();
    size_t left = msg.size();
    while (left > 0) {
        ssize_t n = ::send(fd, p, left, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        left -= static_cast<size_t>(n);
        p += n;
    }
    return true;
}

std::optional<std::pair<std::string,std::string>>
split_once_bar(const std::string& line) {
    auto pos = line.find('|');
    if (pos == std::string::npos) return std::nullopt;
    return std::make_pair(line.substr(0, pos), line.substr(pos + 1));
}

bool parse_csv_doubles(const std::string& csv, std::vector<double>& out) {
    out.clear();
    std::istringstream iss(csv);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        if (tok.empty()) continue;
        try {
            out.push_back(std::stod(tok));
        } catch (...) {
            return false;
        }
    }
    return true;
}

std::string to_csv(const Eigen::Ref<const Eigen::VectorXd>& v) {
    std::ostringstream oss;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << v[i];
    }
    return oss.str();
}

} // namespace

// ---------- Communicator ----------
Communicator::Communicator(int port) : server_fd_(-1) {
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) fail_here_("socket() fail");

    int yes = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::htonl(INADDR_ANY);
    addr.sin_port = ::htons(port);

    if (::bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(server_fd_);
        fail_here_("bind() fail");
    }
    if (::listen(server_fd_, 16) < 0) {
        ::close(server_fd_);
        fail_here_("listen() fail");
    }

    std::cout << "[C++] Communicator ouvindo em 0.0.0.0:" << port << "\n";
}

Communicator::~Communicator() {
    if (server_fd_ >= 0) ::close(server_fd_);
}

void Communicator::serve_forever(IController& controller) {
    while (true) {
        sockaddr_in cli{};
        socklen_t len = sizeof(cli);
        int client_fd = ::accept(server_fd_, (sockaddr*)&cli, &len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            std::cerr << "accept() fail: " << std::strerror(errno) << "\n";
            continue;
        }

        // timeouts básicos
        timeval tv{.tv_sec = 2, .tv_usec = 0};
        ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        std::string line;
        if (!read_line(client_fd, line)) {
            ::close(client_fd);
            continue;
        }

        // Espera "csv(x)|csv(xref)"
        std::vector<double> vx, vxr;
        bool ok = false;
        if (auto both = split_once_bar(line)) {
            ok = parse_csv_doubles(both->first, vx)
              && parse_csv_doubles(both->second, vxr);
        }

        if (ok) {
            // Map sem cópia para Eigen::VectorXd
            Eigen::Map<const Eigen::VectorXd> x(vx.data(),  static_cast<Eigen::Index>(vx.size()));
            Eigen::Map<const Eigen::VectorXd> xr(vxr.data(), static_cast<Eigen::Index>(vxr.size()));

            Eigen::VectorXd u;
            try {
                u = controller.computeU(x, xr);
            } catch (const std::exception& e) {
                (void)e;
                u = Eigen::VectorXd(); // vazio → mandaremos "nan"
            } catch (...) {
                u = Eigen::VectorXd();
            }

            if (u.size() > 0) {
                write_line(client_fd, to_csv(u));
            } else {
                write_line(client_fd, "nan");
            }
        } else {
            write_line(client_fd, "nan");
        }

        ::close(client_fd);
    }
}

[[noreturn]] void Communicator::fail_here_(const char* what) {
    std::cerr << what << ": " << std::strerror(errno) << "\n";
    std::exit(1);
}

