#executable
TARGET = dla_cpu
#source files
SRC = main.cu lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -O2 -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) dla.png