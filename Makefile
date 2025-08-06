#executable
TARGET = dla_cpu
#source files
SRC = main.cu lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -02 -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)