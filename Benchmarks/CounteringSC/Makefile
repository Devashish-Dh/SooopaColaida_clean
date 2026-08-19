COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

SUBDIRS := interwarp_same_value single_thread_discard_fp instrumentation_creates_sync virtual_alias_proxy_fence

.PHONY: all run clean $(SUBDIRS)

all: $(SUBDIRS)

run:
	@set -e; \
	for dir in $(SUBDIRS); do \
		echo ""; \
		echo "=== $$dir ==="; \
		$(MAKE) --no-print-directory -C "$(COMMON_DIR)/$$dir" run; \
	done

clean:
	@set -e; \
	for dir in $(SUBDIRS); do \
		$(MAKE) --no-print-directory -C "$(COMMON_DIR)/$$dir" clean; \
	done

$(SUBDIRS):
	@$(MAKE) --no-print-directory -C "$(COMMON_DIR)/$@"
