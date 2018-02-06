CXX=g++
ED_DIR=/home/abner/Downloads/EDCirclesSrc
INCLUDE_DIRS=-I./ -I$(ED_DIR)
CXX_FLAGS=-std=c++11 $(INCLUDE_DIRS)
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LIBS=$(OPENCV_LIBS) 
LD_FLAGS=$(LIBS)
TARGET=ARHT

HEADERS=$(wildcard *.h)
SOURCES=$(patsubst %.h,%.cpp,$(HEADERS))
OBJECTS=$(patsubst %.h,release/%.o,$(HEADERS)) 
ED_OBJECTS=$(wildcard $(ED_DIR)/release/*.o)

release/%.o: %.cpp %.h
	@mkdir -p release
	$(CXX) $(CXX_FLAGS) -c $< -o $@ 
default: $(TARGET)
clean:
	rm -f $(TARGET) release/*
$(TARGET): $(OBJECTS) main.cpp
	$(CXX) $(CXX_FLAGS) main.cpp -o $(TARGET) $(OBJECTS) $(ED_OBJECTS) $(LD_FLAGS)
all: clean default
