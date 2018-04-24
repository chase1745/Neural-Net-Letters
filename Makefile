OBJ = main.o Neural_Net.o
MAIN = main

$(MAIN): $(OBJ)
	c++-7.2.0 -std=c++17 -o $(MAIN) $(OBJ)

main.o: main.cpp
	c++-7.2.0 -std=c++17 -c main.cpp

Neural_Net.o: Neural_Net.cpp Neural_Net.h
	c++-7.2.0 -std=c++17 -c Neural_Net.cpp

clean:
	rm -f $(OBJ) $(MAIN)
