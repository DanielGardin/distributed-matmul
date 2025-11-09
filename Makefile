CC := gcc
CFLAGS := -Iinclude -Wall -Wextra
SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)
BIN := matmul

.PHONY: all debug mpi clean

all: CFLAGS += -O3
all: $(BIN)

debug: CFLAGS += -g
debug: $(BIN)

mpi: CC := mpicc
mpi: CFLAGS += -DOPENMPI -O3
mpi: LDFLAGS += -lmpi
mpi: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) $(LDFLAGS) -lopenblas -o $@

clean:
	rm -f $(OBJ) $(BIN)
