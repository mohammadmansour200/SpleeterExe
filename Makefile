CXX = g++
CXXFLAGS = -std=c++17 -I./src/spleeter/include -I./third_party/tensorflow-cpu/include
LDFLAGS = -static-libstdc++ -static-libgcc -L./third_party/tensorflow-cpu/lib -ltensorflow -ltensorflow_framework \
          -static-libstdc++ -static-libgcc \
	  -Wl,-rpath='$$ORIGIN' -Wl,-rpath='/usr/lib/Basset'

VERSION = 0.1.0
BUILD_DIR = build
SRC_DIR = src

SOURCES = main.cpp $(SRC_DIR)/spleeter/SpleeterProcessor.cpp

TARGET = $(BUILD_DIR)/SpleeterExe

$(shell mkdir -p $(BUILD_DIR))

$(TARGET): $(SOURCES)
	$(CXX) $(SOURCES) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)

install:
	cp ./third_party/tensorflow-cpu/lib/libtensorflow.so $(BUILD_DIR)/libtensorflow.so.1
	cp ./third_party/tensorflow-cpu/lib/libtensorflow_framework.so $(BUILD_DIR)/libtensorflow_framework.so.1
	cp -r ./models $(BUILD_DIR)/

package:
	tar -czf SpleeterExe_$(VERSION).tar.gz -C $(BUILD_DIR) .

.PHONY: all clean install package

