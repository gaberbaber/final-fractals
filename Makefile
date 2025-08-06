TARGET = 
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -02 -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)