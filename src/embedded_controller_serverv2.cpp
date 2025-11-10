// file: embedded_controller_server.cpp
// build: g++ -O2 -std=c++17 embedded_controller_server.cpp communicator.cpp qpoases_utils.cpp -o embedded_server
// run:   ./embedded_server 5555

#include <iostream>
#include <string>

// === Eigen (dense + sparse) ===
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <qpOASES.hpp>

#include "SparseMatrixB.hpp"
#include "SparseMatrixD.hpp"
#include "bounds.hpp"
#include "mpc_matrices.hpp"

#include "qpoases_utils.hpp"
#include "icontroller.hpp"
#include "communicator.hpp"


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
