1. Overall picture
------------------
This project builds a firm-to-firm production network in several stages.

First, a sampling step chooses firms from a large firm-level dataset and
creates the basic Excel files.

Second, an IPOPT step solves a continuous optimisation model on the initial
unweighted support.

Third, an irreducibility step fixes the edge list so the network becomes
strongly connected and aperiodic.

Fourth, a small Python script prepares the exact input files needed by CPLEX.

Finally, a C program using CPLEX computes the final weighted production
network.


2. Sector tables: IOT.xlsx and Sij.xlsx
---------------------------------------
These two files come from a sector input–output table (“IOT”).

IOT.xlsx is created by taking the original sector input–output table and
dividing each column by its own column total.  
This makes each column sum to 1 while keeping all zeros in place.

Sij.xlsx is created from the same original input–output table, but by
max-normalising:  
- find the largest positive entry in the entire table,  
- divide every positive entry by this value,  
- leave zeros unchanged.

After this, Sij.xlsx has the same zero structure as the original table, all
positive entries lie in (0,1], and at least one entry equals 1.

These two files are used later in the IPOPT and analysis steps. They are not
inputs to the sampling code.


3. Sampling code
----------------
The sampling code is the first main step. It selects firms, assigns sizes and
sector labels, and builds the initial unweighted support.

Input:
- FIRM.csv  
  This is the cleaned firm-level dataset. It contains counts of firms in each
  size category for each sector (SIC). The sampling code reads these counts
  and draws a sample of firms according to the user-specified target.

The sampling code produces:
- FIRM_STRENGTH.xlsx  
  Columns: Firm_ID, Size
- MAPPING.xlsx  
  Columns: Firm_ID, SIC
- an initial unweighted edge list  
  One buyer–seller pair per line (the name of this file is chosen by you).

Examples of running the sampling script:
    python sampling.py --target-firms 50000
    python sampling.py --target-firms ALL


4. IPOPT code
-------------
The IPOPT program takes:
- MAPPING.xlsx
- FIRM_STRENGTH.xlsx
- the unweighted edge list from the sampling step
- IOT.xlsx and Sij.xlsx (the sector tables)

and solves a continuous optimisation problem on the given support.

It produces:
- continuous edge weights
- diagnostics and logs

This step adjusts weights but does not change the support. The final support
used later comes from the irreducibility step.


5. Irreducibility code
----------------------
The irreducibility code takes an edge list and turns it into a strongly
connected and aperiodic directed graph.

Input:
- the edge list from the previous stage

It identifies strongly connected components, connects them, adds minimal extra
links or self-loops to remove periodicity, and preserves the firm ID scheme.

Output:
- edges_irreducible.txt  
  Each line contains: buyer_id seller_id

This file is the definitive support used by the CPLEX solver.


6. weights_input.txt (input builder for CPLEX)
----------------------------------------------
This Python script prepares the exact input files required by the CPLEX code.

Expected files:
- edges_irreducible.txt
- FIRM_STRENGTH.xlsx
- MAPPING.xlsx

Outputs (written to processed_data/):
- network.txt
- money.txt
- firmSector.txt

6.1 network.txt
----------------
Built from edges_irreducible.txt.

One line per buyer:
    buyer_id,supplier_1,supplier_2,...,supplier_k

Suppliers are sorted and deduplicated.

6.2 money.txt
--------------
Built from FIRM_STRENGTH.xlsx.

The script extracts the numeric part of Firm_ID and pairs it with Size:

    firm_id,size

One line per firm.

6.3 firmSector.txt
-------------------
Built from MAPPING.xlsx.

The script extracts numeric firm IDs and numeric sector codes:

    firm_id,sector_id

Rows with missing values after extraction are dropped.

6.4 Running the script
----------------------
Place the three input files in the same directory and run:

    python3 weights_input.txt

This creates:
- processed_data/network.txt
- processed_data/money.txt
- processed_data/firmSector.txt


7. cplex.c (final weighted network)
-----------------------------------
The C program reads:
- network.txt
- money.txt
- firmSector.txt

It:
- rescales firm sizes internally,
- enforces row-stochasticity,
- enforces firm-level inflow tolerances,
- enforces sector-level inflow tolerances,
- controls self-weights,
- and finally minimises the sum of squared edge weights.

Output:
- a weighted edge list, for example:
      edges_weighted.txt

Each line has:
    buyer_id,seller_id,weight


8. Compiling and running cplex.c
--------------------------------
To compile you need a C compiler and CPLEX installed.

The compile command generally includes:
- the compiler with optimisation flags,
- the include directory for CPLEX,
- the library directory for CPLEX,
- linking against CPLEX and system libraries,
- choosing an output executable name.

Example structure:

    <compiler> <flags> cplex.c \
        -I<CPLEX include> \
        -L<CPLEX lib> \
        -lcplex -lm -lpthread -ldl \
        -o <solver_program>

To run:

    <solver_program> network.txt money.txt firmSector.txt <output_file> <parameters>

The resulting file contains:
    buyer_id,seller_id,weight

This is the final weighted production network.

