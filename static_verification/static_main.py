# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("=" * 80)
    print("COMPREHENSIVE VERIFICATION OF LECLERC'S THESIS")
    print("Prospect Theory Equilibrium in Games")
    print("=" * 80)

    # ========================================================================
    # PART 1: IMPLEMENTATION VERIFICATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    checks, pt = verify_pt_implementation()

    print("\nPT Implementation Checks:")
    print("-" * 40)
    for check in checks:
        status = "✓" if check['passed'] else "✗"
        print(f"{status} {check['test']:35} {check['result']}")
        if not check['passed']:
            print(f"  Required: {check['required']}")

    # ========================================================================
    # PART 2: FINE SEARCH FOR OCHS GAME PT-EB
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: FINE SEARCH FOR OCHS GAME PT-EB")
    print("=" * 80)

    ochs_results = fine_search_ochs_game()
    print("\nFine Grid Search Results for Ochs Game:")
    print("-" * 50)
    for res in ochs_results:
        status = "✓" if res['close_to_claim'] else "○"
        print(f"{status} {res['params']:25} p={res['p']:.3f}, q={res['q']:.3f}")
        print(f"  Best responses: br1={res['br1']:.3f}, br2={res['br2']:.3f}")
        print(f"  Deviation: {res['deviation']:.6f}")

    # ========================================================================
    # PART 3: COMPREHENSIVE GAME ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 3: COMPREHENSIVE GAME ANALYSIS")
    print("=" * 80)

    analysis_results = run_comprehensive_analysis()

    print("\nDetailed Game Analysis:")
    print("-" * 80)

    for result in analysis_results:
        print(f"\nGAME: {result['game']}")
        print(f"Description: {result['description']}")
        print(f"PT Parameters: λ={result['pt_params'].get('lambd', 'N/A')}, "
              f"α={result['pt_params'].get('alpha', 'N/A')}, "
              f"γ={result['pt_params'].get('gamma', 'N/A')}, "
              f"r={result['pt_params'].get('r', 'N/A')}")

        print(f"Classical NE:")
        if result['classical_pure']:
            print(f"  Pure: {result['classical_pure']}")
        if result['classical_mixed']:
            p, q = result['classical_mixed']
            print(f"  Mixed: p={p:.3f}, q={q:.3f}")

        print(f"Prospect Theory:")
        if result['pt_ne']:
            p, q = result['pt_ne']
            print(f"  PT-NE: p={p:.3f}, q={q:.3f}")
            if result['classical_mixed']:
                p_class, q_class = result['classical_mixed']
                delta_p = abs(p_class - p)
                delta_q = abs(q_class - q)
                print(f"    vs Classical: Δp={delta_p:.3f}, Δq={delta_q:.3f}")
        else:
            print(f"  PT-NE: NOT FOUND (pathology)")

        if result['pt_eb']:
            p, q = result['pt_eb']
            print(f"  PT-EB: p={p:.3f}, q={q:.3f}")
        else:
            print(f"  PT-EB: Not found")

    # ========================================================================
    # PART 4: THESIS CLAIM VERIFICATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 4: THESIS CLAIM VERIFICATION")
    print("=" * 80)

    claims = verify_thesis_claims(analysis_results)

    print("\nTHESIS CLAIM ASSESSMENT:")
    print("-" * 60)

    supported_count = 0
    total_claims = len(claims)

    for claim_name, claim_data in claims.items():
        status = "✓" if claim_data['supported'] else "✗"
        if claim_data['supported']:
            supported_count += 1

        print(f"\n{status} {claim_data['description']}")
        if claim_data['games']:
            games_str = ", ".join(claim_data['games'])
            print(f"  Supported by: {games_str}")
        else:
            print(f"  Not supported by tested games")

    # ========================================================================
    # PART 5: SUMMARY AND CONCLUSION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 5: SUMMARY AND CONCLUSION")
    print("=" * 80)

    support_rate = supported_count / total_claims * 100

    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Claims supported: {supported_count}/{total_claims} ({support_rate:.1f}%)")

    if support_rate >= 75:
        conclusion = "Thesis claims are LARGELY SUPPORTED"
    elif support_rate >= 50:
        conclusion = "Thesis claims are PARTIALLY SUPPORTED"
    else:
        conclusion = "Thesis claims are LARGELY UNSUPPORTED"

    print(f"  CONCLUSION: {conclusion}")

    print(f"\nKEY FINDINGS:")
    print(f"  1. PT implementation verified: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    print(f"  2. Ochs game PT-EB search: Found candidates close to thesis claim" if
          any(r['close_to_claim'] for r in ochs_results) else
          "  2. Ochs game PT-EB search: No exact match found")

    # Specific findings
    print(f"\nSPECIFIC RESULTS:")
    for result in analysis_results:
        if result['game'] == 'OchsGame_Correct' and result['pt_ne'] is None:
            print(f"  ✓ Ochs Game shows PT-NE failure (as thesis claims)")
        if result['game'] == 'CrawfordGame' and result['pt_vs_classical'] == 'Different':
            print(f"  ✓ Crawford Game shows PT changes predictions")
        if result['game'] == 'BattleOfSexes' and result['pt_vs_classical'] == 'Different':
            print(f"  ✓ Battle of Sexes shows PT changes predictions")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


# ============================================================================
# 8. RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    main()

    # After displaying results
    print("\n" + "="*80)
    run_diag = input("Run advanced diagnostics? (y/n): ").strip().lower()
    if run_diag == 'y':
        run_advanced_diagnostics()
