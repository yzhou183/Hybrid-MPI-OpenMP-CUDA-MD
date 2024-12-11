
#### 6. `Makefile`
```makefile
CC = mpic++
CXXFLAGS = -std=c++11 -Iinclude -g -Wall
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
SRC_DIR = src
OBJ_DIR = obj
TEST_DIR = tests
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SOURCES))

EXECUTABLE = $(BIN_DIR)/particle_simulation
TEST_EXECUTABLE = $(BIN_DIR)/test_particle

$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

all: $(EXECUTABLE) $(TEST_EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(TEST_EXECUTABLE): $(TEST_OBJECTS)
	g++ $(CXXFLAGS) -o $@ $^ -lgtest -lgtest_main

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	g++ $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(BIN_DIR)/*
