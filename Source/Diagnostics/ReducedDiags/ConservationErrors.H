/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CONSERVATIONERRORS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CONSERVATIONERRORS_H_

#include "ReducedDiags.H"

#include <string>

/**
 * Diagnostic that writes per-step total charge, energy and momentum
 * (particle + field contributions) to one CSV file.
 */
class ConservationErrors : public ReducedDiags
{
public:
    ConservationErrors (const std::string& rd_name);

    void ComputeDiags (int step) final;
};

#endif
