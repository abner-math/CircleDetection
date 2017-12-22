CXX=g++
INCLUDE_DIRS=
CXX_FLAGS=-std=c++11 $(INCLUDE_DIRS)
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LIBS=$(OPENCV_LIBS) 
LD_FLAGS=$(LIBS)
TARGET=ARHT 

default: $(TARGET)
$(TARGET): main.cpp
	$(CXX) $(CXX_FLAGS) main.cpp -o $(TARGET) $(LD_FLAGS)
