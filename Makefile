CC=g++
all : train predict
train : train.o
	$(CC) -o train train.o

predict : predict.o
	$(CC) -o predict predict.o

train.o: train.cpp
	$(CC) -c train.cpp
predict.o: predict.cpp
	$(CC) -c predict.cpp

clean:
	rm -rf *.o
	rm -rf train predict

