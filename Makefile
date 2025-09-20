# Makefile - CBC Population Distributions
# ======================================
# Usage:
#   make run        # download input (if needed) + run sampler
#   make clean      # delete chunked JSONs, keep *_all.json / *_all.h5

# ---- Config (override with: make run NSAMPLES=10000 OUTDIR=out ect.) ----

# GWTC-4 hyperparameters file (Broken Power Law + Two Peaks model).
HYP_FILE ?= O1O2O3all_mass_powerlaw_redshift_maxP_events_all.h5
HYP_URL  ?= https://dcc.ligo.org/public/$(HYP_FILE)

OUTDIR   ?= output
NSAMPLES ?= 1000000
CHUNK    ?= 1000
ZMAX     ?= 2.3
PAIRING  ?= 1   # 1=pairing (default), 0=independent

.PHONY: all download run clean

all: run

# Download the hyperparameters file if missing
download: $(HYP_FILE)
$(HYP_FILE):
	curl -fL -o "$@" "$(HYP_URL)"

# Force re-download of the hyperparameters file, if needed
update:
	curl -fL -o "$(HYP_FILE)" "$(HYP_URL)"

# Run CBC sampler (calls cbc_population_distributions/population_driver.py via CLI)
run: download
	uv run cbc-sample \
	  --hyperparams-file "$(HYP_FILE)" \
	  --outdir "$(OUTDIR)" \
	  --n-samples $(NSAMPLES) \
	  --chunk-size $(CHUNK) \
	  --z-max $(ZMAX) \
	  $(if $(filter 1,$(PAIRING)),--pairing,--no-pairing)

# Clean only chunked JSONs (keep final *_all.json and *_all.h5 outputs)
clean:
	rm -f "$(OUTDIR)"/*_[0-9].json
