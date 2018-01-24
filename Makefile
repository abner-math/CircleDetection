CXX=g++
INCLUDE_DIRS=-I./
CXX_FLAGS=-std=c++11 $(INCLUDE_DIRS)
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LIBS=$(OPENCV_LIBS) 
LD_FLAGS=$(LIBS)
TARGET=ARHT 

HEADERS=$(wildcard *.h)
SOURCES=$(patsubst %.h,%.cpp,$(HEADERS))
OBJECTS=$(patsubst %.h,release/%.o,$(HEADERS)) 

release/%.o: %.cpp %.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@ 
default: $(TARGET)
$(TARGET): $(OBJECTS) main.cpp
	$(CXX) $(CXX_FLAGS) main.cpp -o $(TARGET) $(LD_FLAGS) $(OBJECTS)
