"""
Temporal outbreak engine (Handoff 9).

Runs HIV transmission on the time-ordered event stream from the mobility
generator.  The static contact graph stays for structure validation;
the outbreak consumes the raw event stream.

ETHICAL INVARIANT: Ensemble-from-distributions only.  No real individual
trajectories.  HIV held out of all structural decisions.
"""
