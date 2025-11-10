#pragma once

// forward-declare to avoid including eigen here
struct IController;

class Communicator {
public:
    explicit Communicator(int port);
    ~Communicator();

    // não copiável
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    void serve_forever(IController& controller);

private:
    int server_fd_;
    [[noreturn]] static void fail_here_(const char* what);
};

