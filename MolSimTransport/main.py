# MolSimTransport/main.py
import argparse

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="MolSimTransport: A python package for quickly calculating the transport properties of molecular junctions at multiple scales",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add subparsers for each module
    subparsers = parser.add_subparsers(dest="module", help="Available modules\n\n")
    # Module: L1_EHT
    parser_eht = subparsers.add_parser(
        "L1_EHT", help="Calculate the transport properties of isolated molecule based on the Extended Hückel Theory (EHT) method\n\n"
    )
    # Module: L1_XTB
    parser_xtb = subparsers.add_parser(
        "L1_XTB", help="Calculate the transport properties of isolated molecule based on the xTB method\n\n"
    )
    # Module: L2_Align
    parser_l2_align = subparsers.add_parser(
        "L2_Align", help="Align the user-built Extended Molecule (EM) with the supplied electrode clusters\n\n"
    )
    # Module: L2_Trans
    parser_l2_trans = subparsers.add_parser(
        "L2_Trans", help="Calculate the transport properties of \"extended molecule + cluster electrode\" based on the xTB method\n\n" 
    )
    # Module: L2_MPSH
    parser_l2_mpsh = subparsers.add_parser(
        "L2_MPSH", help="Calculate the MPSH (Molecular Projected Self-Consistent Hamiltonian) and generate the Molden file in the L2 scheme\n\n"
    )
    # Module: L3_Trans
    parser_l3_trans = subparsers.add_parser(
        "L3_Trans", help="Calculate the transport properties of molecular junction containing the principal layer (PL) of the electrode based on the xTB method\n\n"
    )
    # Module: L3_EEF
    parser_l3_eef = subparsers.add_parser(
        "L3_EEF", help="Calculate the transport properties of molecular junction containing the PL under the electric field\n\n"
    )
    # Module: L3_MPSH
    parser_l3_mpsh = subparsers.add_parser(
        "L3_MPSH", help="Calculate the MPSH and generate the Molden file in the L3 scheme\n\n"
    )
    # Module: L3_EC
    parser_l3_ec = subparsers.add_parser(
        "L3_EC", help="Analyze the eigenstate at a specific energy point and generate the corresponding Molden file for the EigenChannel\n\n"
    )
    # Parse arguments
    args = parser.parse_args()

    # If no module is specified, display the help information
    if not args.module:
        parser.print_help()

if __name__ == "__main__":
    main()
