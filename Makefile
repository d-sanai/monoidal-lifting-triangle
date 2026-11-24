# Monoidal Lifting Triangles (MLT) — One-Command Build & Verification Suite
# Zero-heap, -lm-free C99 core + Shared Library + Python Zero-Copy Benchmarks

CC       ?= cc
PYTHON   ?= python3
CFLAGS   := -std=c99 -O3 -Wall -Wextra -I. -Iinclude
LDFLAGS  :=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHLIB_EXT   := dylib
    SHLIB_FLAGS := -dynamiclib -fPIC
else
    SHLIB_EXT   := so
    SHLIB_FLAGS := -shared -fPIC
    LDFLAGS     += -lpthread
endif

BUILD_DIR := build

# Auto-discover all headers and C99 sources in root (./) and benchmarks/
HDR_FILES  := $(wildcard *.h include/*.h)
ALL_C_SRCS := $(wildcard *.c benchmarks/*.c)
FFI_SRCS   := $(filter %_ffi.c,$(ALL_C_SRCS)) $(wildcard src/*_ffi.c)
BENCH_SRCS := $(filter-out %_ffi.c,$(ALL_C_SRCS))
BENCH_BINS := $(sort $(patsubst %.c,$(BUILD_DIR)/%,$(notdir $(BENCH_SRCS))))

SHLIB_TARGET := liblifting_core_v2.$(SHLIB_EXT)
PY_TESTS     := $(wildcard *.py benchmarks/*.py)

vpath %.c . benchmarks src

.PHONY: all test bench shlib pytest paper clean help

all: bench shlib

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.c $(HDR_FILES) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bench: $(BENCH_BINS)
ifeq ($(strip $(BENCH_SRCS)),)
	@echo "No .c benchmark files found in ./ or benchmarks/."
	@echo "Copy lifting_core_v2.h and your *.c files into ./ or benchmarks/ and re-run."
else
	@echo "=== Built $(words $(BENCH_BINS)) C99 benchmark binaries in $(BUILD_DIR)/ ==="
endif

shlib: $(FFI_SRCS) $(HDR_FILES)
ifneq ($(strip $(FFI_SRCS)),)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(FFI_SRCS) -o $(SHLIB_TARGET) $(LDFLAGS)
	@if [ -d benchmarks ]; then cp $(SHLIB_TARGET) benchmarks/; fi
	@echo "=== Built $(SHLIB_TARGET) ==="
endif

test: bench
ifeq ($(strip $(BENCH_BINS)),)
	@exit 1
endif
	@echo "=================================================================="
	@echo " Running All C99 Exact 0-Bit Verification Suites"
	@echo "=================================================================="
	@for bin in $(BENCH_BINS); do \
		echo ""; \
		echo ">>> Executing $$bin <<<"; \
		./$$bin || exit 1; \
	done
	@echo ""
	@echo "=================================================================="
	@echo " ALL $(words $(BENCH_BINS)) C99 BENCHMARKS PASSED WITH 0 ERRORS"
	@echo "=================================================================="

pytest: shlib
	@for py in $(PY_TESTS); do \
		echo ""; \
		echo ">>> Executing $(PYTHON) $$py <<<"; \
		PYTHONPATH=benchmarks:. $(PYTHON) $$py || exit 1; \
	done

paper:
	@if [ -f paper/main.tex ]; then \
		cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; \
	elif [ -f main.tex ]; then \
		pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; \
	else \
		echo "No main.tex found."; \
	fi

clean:
	rm -rf $(BUILD_DIR) $(SHLIB_TARGET) benchmarks/$(SHLIB_TARGET) *.aux *.log *.out paper/*.aux paper/*.log paper/*.out

help:
	@echo "Available targets:"
	@echo "  make test    - Build and run all standalone C99 verification suites"
	@echo "  make shlib   - Build shared library ($(SHLIB_TARGET)) for Python ctypes/NumPy"
	@echo "  make pytest  - Run Python zero-copy NumPy and continual learning benchmarks"
	@echo "  make paper   - Compile LaTeX manuscript (main.tex) to PDF"
	@echo "  make clean   - Remove build artifacts"