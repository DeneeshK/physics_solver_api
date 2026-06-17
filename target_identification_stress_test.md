# Target-identification stress test

These all share the structural pattern that broke: several sentences of
setup heavily describing OTHER quantities, with the actual ask stated
explicitly at the very end, asking for something different from what
dominates the sentence. Run each 2-3 times — temperature is 0.1, not 0,
so some sampling variance is expected, but the unknown.symbol it picks
should be consistent across repeats now.

For each, check `answer.symbol` matches what's in [brackets] and that
`chain_summary` has more than one step (i.e. it didn't stop early at an
intermediate quantity the way the original bug did).

1. "A body of density 8000 kg/m³ and volume 0.5 m³ accelerates from
   10 m/s to 30 m/s over 40 m. Find the net force." [F — the original
   failing case]

2. "A car travels at 20 m/s and decelerates uniformly, covering 100 m
   before coming to rest. Calculate the time taken to stop." [t — setup
   dominated by velocity/distance, ask is for time]

3. "A resistor of 50 ohms carries a current that produces a potential
   difference of 10 V across it, in a circuit where the current flows
   for 2 minutes. Determine the power dissipated in the resistor."
   [P — setup dominated by resistance/voltage/duration, ask is for power]

4. "A satellite orbits at a radius of 7000 km with an orbital period of
   5800 seconds. What is the mass of the planet it orbits?" [M — setup
   dominated by radius/period, ask is for the central mass]

5. "A block of mass 2 kg is raised to a height of 5 m and then released,
   falling freely under gravity. Find the velocity of the block just
   before it hits the ground." [v — setup dominated by mass/height, ask
   is for final velocity, which itself isn't the most natural-sounding
   "answer" given how much the sentence talks about height and falling]
