// file: embedded_controller_server.cpp
// build: g++ -O2 -std=c++17 embedded_controller_server.cpp -o embedded_server
// run:   ./embedded_server 5555

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <optional>

// === Eigen (dense + sparse) ===
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <qpOASES.hpp>

#include "SparseMatrixB.hpp"
#include "SparseMatrixD.hpp"
#include "bounds.hpp"
#include "mpc_matrices.hpp"

// A: ponteiro para dados row-wise, com tamanho rows*cols
inline void print_matriz_qpoases(const qpOASES::real_t* A,
                                 std::size_t rows,
                                 std::size_t cols)
{
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << A[r * cols + c];
            if (c + 1 < cols) std::cout << ' ';
        }
        std::cout << '\n';
    }
}

// Converte uma Eigen::SparseMatrix<double> para um buffer DENSO row-wise de qpOASES::real_t.
// Retorna ponteiro alocado com malloc (use std::free depois).
inline qpOASES::real_t* to_rowwise_dense(const Eigen::SparseMatrix<double>& B) {
    const int m = B.rows();
    const int n = B.cols();

    const size_t len = static_cast<size_t>(m) * static_cast<size_t>(n);
    qpOASES::real_t* buf = static_cast<qpOASES::real_t*>(
        std::malloc(len * sizeof(qpOASES::real_t))
    );
    if (!buf) throw std::bad_alloc{};

    // zera tudo (entradas implícitas do sparse viram 0)
    std::fill_n(buf, len, static_cast<qpOASES::real_t>(0));

    // (opcional) B.makeCompressed();

    // copia somente os não-nulos para as posições row-wise corretas
    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            buf[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)]
                = static_cast<qpOASES::real_t>(it.value());
        }
    }

    return buf; // lembre-se de std::free(buf)
}

// Converte um Eigen::VectorXd para um buffer contíguo de qpOASES::real_t.
// Retorna ponteiro alocado com malloc (use std::free depois).
inline qpOASES::real_t* to_buffer(const Eigen::VectorXd& v) {
    const Eigen::Index n = v.size();
    qpOASES::real_t* buf = static_cast<qpOASES::real_t*>(
        std::malloc(static_cast<size_t>(n) * sizeof(qpOASES::real_t))
    );
    if (!buf) throw std::bad_alloc{};

    for (Eigen::Index i = 0; i < n; ++i) {
        buf[i] = static_cast<qpOASES::real_t>(v[i]);
    }
    return buf; // lembre-se de std::free(buf)
}

// ---------------- IO helpers (line-based) ----------------
static bool read_line(int fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
        ssize_t n = ::recv(fd, &c, 1, 0);
        if (n == 0) return false;           // peer fechou
        if (n < 0) {
            if (errno == EINTR) continue;   // sinal; tenta de novo
            return false;
        }
        if (c == '\n') break;
        if (c == '\r') continue;            // tolera CRLF
        out.push_back(c);
        if (out.size() > 2'000'000) return false; // proteção contra linha gigante
    }
    return true;
}

static bool write_line(int fd, const std::string& s) {
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

// ---------------- CSV helpers ----------------
static std::optional<std::pair<std::string,std::string>>
split_once_bar(const std::string& line) {
    auto pos = line.find('|');
    if (pos == std::string::npos) return std::nullopt;
    return std::make_pair(line.substr(0, pos), line.substr(pos + 1));
}

static bool parse_csv_doubles(const std::string& csv, std::vector<double>& out) {
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

static std::string to_csv(const Eigen::Ref<const Eigen::VectorXd>& v) {
    std::ostringstream oss;
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << v[i];
    }
    return oss.str();
}

// ---------------- Controller interface (Eigen-friendly) ----------------
struct IController {
    virtual ~IController() = default;
    // Usar Ref evita cópia e permite receber Map ou blocos de VectorXd
    virtual Eigen::VectorXd computeU(const Eigen::Ref<const Eigen::VectorXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& xref) = 0;
};

// Implementação dummy: NÃO faz contas; retorna um vetor de zeros de tamanho nu
struct DummyController final : IController {
    explicit DummyController(Eigen::Index nu, Eigen::Index nx, Eigen::Index nh)
        : nu_(nu), nx_(nx), nh_(nh)
    {
        // Exemplo de estrutura para você usar depois:
        // K_ é esparsa (nu x nx). Aqui limpinha, sem setar nada.
        auto B = makeSparseMatrixB();
        auto D = makeSparseMatrixD();

	Eigen::VectorXd b = Eigen::VectorXd::Zero(nh_*(nx_+nu_));
	Eigen::VectorXd lbA = makeLB();
	Eigen::VectorXd ubA = makeUB();

	qpOASES::real_t* B_dense = to_rowwise_dense(B);
	qpOASES::real_t* D_dense = to_rowwise_dense(D);

	qpOASES::real_t* b_dense = to_buffer(b);
	qpOASES::real_t* lbA_dense = to_buffer(lbA);
	qpOASES::real_t* ubA_dense = to_buffer(ubA);
	
	B_ = B;
	D_ = B;
	b_ = b;
	lbA_ = lbA;
	ubA_ = ubA;

	// Converter para denso row-wise
	B_dense_ = B_dense;
	D_dense_ = D_dense;
	b_dense_ = b_dense;
	lbA_dense_ = lbA_dense;
	ubA_dense_ = ubA_dense;
	std::cout << nh_ * (nx_ + nu_) <<std::endl;
	qp_ = qpOASES::QProblem(nh_*(nx_+nu_), nh_*(nx_+nu_));
	
	qpOASES::Options options;
	options.printLevel = qpOASES::PL_LOW;
	qp_.setOptions(options);

	int nWSR = 10000;
  	qpOASES::returnValue rv = qp_.init(B_dense, b_dense, D_dense, nullptr, nullptr, lbA_dense, ubA_dense, nWSR);
  	if (rv != qpOASES::SUCCESSFUL_RETURN) {
    		std::cerr << "qpOASES: init falhou (code " << rv << ")\n";
  	}
	
	// AAA TESTAR SE A SOLUCAO TA BATENDO COM O QUE TEM LA!
	//qpOASES::real_t z[320];
	//qp_.getPrimalSolution(z);	
		
	//print_matriz_qpoases(B_dense, ( nx_ + nu_ ) * nh_, ( nx_ + nu_) * nh_);
	Fr_ = makeF_r();
	Q_ = makeQ();
	P_ = makeP();
    }

    Eigen::VectorXd computeU(const Eigen::Ref<const Eigen::VectorXd>& x,
                             const Eigen::Ref<const Eigen::VectorXd>& xref) override
    {
        std::cerr << "[C++] computeU: x=" << x.size() << " xref=" << xref.size() << "\n";
	// Garanta dimensão mínima esperada; ajuste como preferir
        if (x.size() != nx_ || xref.size() != nx_*nh_) {
            // Se dimensões não baterem, devolve zeros no tamanho nu_
            return Eigen::VectorXd::Zero(nu_);
        }
	
        // ====== LUGAR PARA SUA LÓGICA REAL ======
        // Exemplo (comentado) de como ficaria usando matriz esparsa:
        //
        // Eigen::VectorXd dx = x - xref;
        // Eigen::VectorXd u  = - (K_ * dx);  // K_ é Eigen::SparseMatrix<double>
        // return u;
	auto Frx = - (Fr_ * x);
	for(int i = 0; i < nx_; i++) {
		lbA_dense_[i] = Frx[i];
		ubA_dense_[i] = Frx[i];
	}
	 
	if (nh_ >= 1) {
	    // j = 0 .. nh_-2  → usa Q
	    for (int j = 0; j < nh_ - 1; ++j) {
	        const int b_off = nu_ + j * (nx_ + nu_);
	        const int x_off = j * nx_;
	
	        // pega o trecho xref[j]
	        const auto xj = xref.segment(x_off, nx_);
	
	        // multiplica e escreve em b_dense_
	        Eigen::VectorXd tmp = Q_ * xj;
	        for (int i = 0; i < nx_; ++i) {
	            b_dense_[b_off + i] = -tmp[i];
	        }
	    }
	
	    // último bloco (j = nh_-1) → usa P
	    const int j_last = nh_ - 1;
	    const int b_off_last = nu_ + j_last * (nx_ + nu_);
	    const int x_off_last = j_last * nx_;
	
	    const auto xN = xref.segment(x_off_last, nx_);
	    Eigen::VectorXd tmpN = P_ * xN;
	    for (int i = 0; i < nx_; ++i) {
	        b_dense_[b_off_last + i] = -tmpN[i];
	    }
	}

	int nWSR = 10000;
	qp_.hotstart(b_dense_, nullptr, nullptr, lbA_dense_, ubA_dense_, nWSR);
	qpOASES::real_t z[320];
	
	qp_.getPrimalSolution(z);

	//print_matriz_qpoases(lbA_dense_, ( nx_ + nu_ ) * nh_, 1);
	//print_matriz_qpoases(ubA_dense_, ( nx_ + nu_ ) * nh_, 1);

	//std::cout << "Solucao: " << z[0] << ", " << z[1] << ", " << z[2] << ", " << z[3] <<  std::endl;
	
	//std::cout << "Press Enter to continue..." << std::flush;
    	//std::cin.get();             // waits for '\n'
	
	//std::cout << "x: " << - (Fr_.transpose() * x) << std::endl;
	//std::cout << "z: " << z << std::endl;
	//std::cout << "xref: " << xref << std::endl;

        Eigen::VectorXd output = (Eigen::VectorXd(4) << z[0], z[1], z[2], z[3]).finished();
        return output;
    }

public:
    Eigen::Index nu_{4};  // tamanho de u (ex.: 4 hélices)
    Eigen::Index nx_{12}; // tamanho de x/xref, ajuste ao seu caso
    Eigen::Index nh_{20}; // tamanho de x/xref, ajuste ao seu caso
    Eigen::SparseMatrix<double> B_; // exemplo de ganho esparso
    Eigen::SparseMatrix<double> D_; // exemplo de ganho esparso
    Eigen::VectorXd b_; // exemplo de ganho esparso
    Eigen::VectorXd lbA_; // exemplo de ganho esparso
    Eigen::VectorXd ubA_; // exemplo de ganho esparso
    
    qpOASES::real_t* B_dense_; // exemplo de ganho esparso
    qpOASES::real_t* D_dense_; // exemplo de ganho esparso
    qpOASES::real_t* b_dense_; // exemplo de ganho esparso
    qpOASES::real_t* lbA_dense_; // exemplo de ganho esparso
    qpOASES::real_t* ubA_dense_; // exemplo de ganho esparso
	
    Eigen::Matrix<double, 12, 12> Fr_;
    Eigen::Matrix<double, 12, 12> P_;
    Eigen::Matrix<double, 12, 12> Q_;
    qpOASES::QProblem qp_;
    

};

// ---------------- Communicator ----------------
class Communicator {
public:
    explicit Communicator(int port) : server_fd_(-1) {
        server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) fail_here_("socket() fail");

        int yes = 1;
        ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = ::htonl(INADDR_ANY); // 0.0.0.0
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

    ~Communicator() {
        if (server_fd_ >= 0) ::close(server_fd_);
    }

    void serve_forever(IController& controller) {
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

private:
    int server_fd_;

    [[noreturn]] static void fail_here_(const char* what) {
        std::cerr << what << ": " << std::strerror(errno) << "\n";
        std::exit(1);
    }
};

// ---------------- main ----------------
int main(int argc, char** argv) {
    int port = 5555;
    if (argc >= 2) port = std::stoi(argv[1]);

    Communicator comm(port);

    // Ajuste nx (dimensão de estado) e nu (dimensão de controle) ao seu caso:
    const Eigen::Index nx = 12; // exemplo
    const Eigen::Index nu = 4;  // exemplo
    const Eigen::Index nh = 20;  // exemplo
    DummyController ctrl(nu, nx, nh); // troque depois por sua implementação real

    comm.serve_forever(ctrl);
    // FALTA DAR FREE NAS COISAS MALOCADA
    // std::free(B_dense);
    // std::free(D_dense);
    // std::free(b_dense);
    // std::free(lbA_dense);
    // std::free(ubA_dense);
    //
    return 0;
}
