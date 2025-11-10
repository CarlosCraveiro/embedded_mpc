all:
	g++ -O2 -lqpOASES -lonnxruntime -std=c++17 src/embedded_controller_serverv2.cpp src/communicator.cpp src/qpoases_utils.cpp -o bin/embedded_server

clean:
	rm -rf bin/embedded_server
