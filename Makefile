all:
	g++ -O2 -lqpOASES -lonnxruntime -std=c++17 src/embedded_controller_serverv2.cpp -o bin/embedded_server

clean:
	rm -rf bin/embedded_server
