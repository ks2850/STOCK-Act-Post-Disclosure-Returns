"""
add_power_committee.py
======================
Augments capitol_trades.csv with a PowerCommittee indicator column.

Power Committees (following Cherry et al. 2017 / Eggers & Hainmueller 2013):
  House:  Appropriations, Ways & Means, Armed Services, Energy & Commerce
  Senate: Appropriations, Finance, Armed Services, Commerce/Science/Transportation

The mapping is Congress-specific (118th: Jan 2023 – Jan 2025; 119th: Jan 2025 – present).
Each trade is matched to the correct Congress based on its traded_date.

SOURCES:
  - 119th Senate Appropriations: senate.gov/general/committee_membership (fetched Mar 2026)
  - 119th Senate assignments: senate.gov/general/committee_assignments/assignments.htm
  - 118th Senate Appropriations: appropriations.senate.gov (Murray/Collins rosters announcement)
  - 118th/119th House Appropriations: Wikipedia, appropriations.house.gov
  - 118th/119th House committees: Wikipedia, official committee pages
  - 117th assignments: govinfo.gov (used for cross-reference)

IMPORTANT: Some assignments may need verification. Run with --audit flag to see
which dataset politicians are NOT matched to any committee. Verify edge cases
against congress.gov or the unitedstates/congress-legislators GitHub repo.

Usage:
  python add_power_committee.py                  # generate augmented CSV
  python add_power_committee.py --audit          # print audit report
  python add_power_committee.py --verify NAME    # check a specific legislator
"""

import csv
import sys
from datetime import datetime

# ===========================================================================
# POWER COMMITTEE MEMBERSHIP MAPPINGS
# Key: (congress_number, chamber, committee_name) -> set of legislator names
# Names must match the politician_name field in capitol_trades.csv
# ===========================================================================

POWER_COMMITTEES = {
    # =====================================================================
    # 118th CONGRESS (Jan 3, 2023 – Jan 3, 2025)
    # =====================================================================

    # --- HOUSE APPROPRIATIONS (118th) ---
    # Source: Wikipedia, appropriations.house.gov, subcommittee rosters
    (118, "House", "Appropriations"): {
        "Tom Cole", "Hal Rogers", "Robert Aderholt", "Mike Simpson",
        "Chuck Fleischmann", "David Joyce", "Guy Reschenthaler",
        "Dan Newhouse", "Stephanie Bice", "Scott Franklin",
        "Michael Guest", "Mike Garcia",
        # Democrats
        "Chellie Pingree", "Debbie Wasserman Schultz", "Ed Case",
        "Julia Letlow", "Lois Frankel", "John Rutherford",
        "Susie Lee", "Debbie Lesko",
    },

    # --- HOUSE WAYS & MEANS (118th) ---
    # Source: Wikipedia, waysandmeans.house.gov
    (118, "House", "Ways and Means"): {
        "Adrian Smith", "Mike Kelly", "David Kustoff", "Ron Estes",
        "Kevin Hern", "Brad Schneider", "French Hill",
        "Terri Sewell", "Suzan DelBene", "Lloyd Doggett",
        "Linda Sanchez", "Judy Chu", "Don Beyer",
        "Earl Blumenauer", "Brad Sherman",
    },

    # --- HOUSE ARMED SERVICES (118th) ---
    # Source: Wikipedia, armedservices.house.gov
    (118, "House", "Armed Services"): {
        "Adam Smith", "Rob Wittman", "Mike Garcia", "Doug Lamborn",
        "Austin Scott", "Michael Waltz", "Mark Green",
        "Scott DesJarlais", "Don Beyer", "Rick Larsen",
        "Seth Moulton", "Mikie Sherrill", "Pat Ryan",
        "Jared Moskowitz", "Jim Banks", "Dan Crenshaw",
        "Mike Flood", "Dale Strong", "Rich McCormick",
        "John James", "Gus Bilirakis",
    },

    # --- HOUSE ENERGY & COMMERCE (118th) ---
    # Source: Wikipedia, energycommerce.house.gov
    (118, "House", "Energy and Commerce"): {
        "Cathy McMorris Rodgers", "Bob Latta", "Gus Bilirakis",
        "Michael Burgess", "Buddy Carter", "Neal Dunn",
        "Diana Harshbarger", "Debbie Lesko", "Greg Steube",
        "Rich McCormick", "Debbie Dingell", "Kathy Castor",
        "Doris Matsui", "Scott Peters", "Sean Casten",
        "Nanette Barragán", "Ann Wagner", "Darrell Issa",
        "Laurel Lee", "Rick Allen",
    },

    # --- SENATE APPROPRIATIONS (118th) ---
    # Source: appropriations.senate.gov (Murray/Collins 118th rosters)
    (118, "Senate", "Appropriations"): {
        # Republicans (Minority in 118th)
        "Susan Collins", "Mitch McConnell", "Lindsey Graham",
        "Jerry Moran", "John Boozman", "Shelley Moore Capito",
        "John Neely Kennedy", "Bill Hagerty", "Katie Britt",
        "Marco Rubio", "Pete Ricketts",
        # Democrats (Majority in 118th)
        "Chris Coons", "Gary Peters", "Joe Manchin",
    },

    # --- SENATE FINANCE (118th) ---
    # Source: finance.senate.gov, Wikipedia
    (118, "Senate", "Finance"): {
        # Democrats (Majority in 118th)
        "Ron Wyden", "Michael Bennet", "Mark Warner",
        "Sheldon Whitehouse", "Tom Carper", "Tina Smith",
        # Republicans (Minority in 118th)
        "Rick Scott", "Bill Hagerty",
        # Note: Ted Cruz is Commerce, NOT Finance
    },

    # --- SENATE ARMED SERVICES (118th) ---
    # Source: armed-services.senate.gov
    (118, "Senate", "Armed Services"): {
        "Dan Sullivan", "Tommy Tuberville", "Markwayne Mullin",
        "Rick Scott", "Katie Britt",
        "Richard Blumenthal", "Tammy Duckworth",
        "Angus King", "Mark Warner",
    },

    # --- SENATE COMMERCE, SCIENCE & TRANSPORTATION (118th) ---
    # Source: commerce.senate.gov, Wikipedia
    (118, "Senate", "Commerce"): {
        "Ted Cruz", "Cynthia Lummis", "Dan Sullivan",
        "Jerry Moran", "Shelley Moore Capito",
        "John Neely Kennedy", "Rick Scott",
        # Democrats
        "Tammy Duckworth", "Gary Peters",
        "Richard Blumenthal", "John Hickenlooper",
        "Tammy Duckworth", "Tina Smith",
        "John Fetterman",
    },

    # =====================================================================
    # 119th CONGRESS (Jan 3, 2025 – present)
    # =====================================================================

    # --- HOUSE APPROPRIATIONS (119th) ---
    # Source: appropriations.house.gov, DeLauro roster announcement Jan 2025
    (119, "House", "Appropriations"): {
        "Tom Cole", "Hal Rogers", "Robert Aderholt", "Mike Simpson",
        "Chuck Fleischmann", "David Joyce", "Guy Reschenthaler",
        "Dan Newhouse", "Stephanie Bice", "Scott Franklin",
        "Michael Guest", "Julia Letlow",
        # Democrats
        "Chellie Pingree", "Debbie Wasserman Schultz", "Ed Case",
        "Lois Frankel", "Susie Lee", "John Rutherford",
    },

    # --- HOUSE WAYS & MEANS (119th) ---
    # Source: waysandmeans.house.gov
    (119, "House", "Ways and Means"): {
        "Adrian Smith", "Mike Kelly", "David Kustoff", "Ron Estes",
        "Kevin Hern", "Brad Schneider", "French Hill",
        "Terri Sewell", "Suzan DelBene", "Lloyd Doggett",
        "Linda Sanchez", "Judy Chu", "Don Beyer",
    },

    # --- HOUSE ARMED SERVICES (119th) ---
    # Source: armedservices.house.gov
    (119, "House", "Armed Services"): {
        "Adam Smith", "Rob Wittman", "Mike Garcia", "Doug Lamborn",
        "Austin Scott", "Mark Green",  # Mark Green resigned Jul 2025
        "Scott DesJarlais", "Don Beyer", "Rick Larsen",
        "Seth Moulton", "Mikie Sherrill", "Pat Ryan",
        "Dan Crenshaw", "Mike Flood", "Dale Strong",
        "Rich McCormick", "John James",
    },

    # --- HOUSE ENERGY & COMMERCE (119th) ---
    # Source: energycommerce.house.gov
    (119, "House", "Energy and Commerce"): {
        "Bob Latta", "Gus Bilirakis", "Michael Burgess",
        "Buddy Carter", "Neal Dunn", "Diana Harshbarger",
        "Greg Steube", "Rich McCormick",
        "Debbie Dingell", "Kathy Castor", "Doris Matsui",
        "Scott Peters", "Sean Casten", "Nanette Barragán",
        "Ann Wagner", "Darrell Issa", "Laurel Lee", "Rick Allen",
    },

    # --- SENATE APPROPRIATIONS (119th) ---
    # Source: senate.gov/general/committee_membership (fetched Mar 28, 2026)
    (119, "Senate", "Appropriations"): {
        # Republicans (Majority)
        "Susan Collins", "Mitch McConnell", "Lindsey Graham",
        "Jerry Moran", "John Boozman", "Shelley Moore Capito",
        "John Neely Kennedy", "Bill Hagerty", "Katie Britt",
        "Markwayne Mullin",
        # Democrats (Minority)
        "Chris Coons", "Gary Peters",
    },

    # --- SENATE FINANCE (119th) ---
    # Source: finance.senate.gov
    (119, "Senate", "Finance"): {
        "Ron Wyden", "Michael Bennet", "Mark Warner",
        "Sheldon Whitehouse", "Tina Smith",
        # Republicans
        "Rick Scott", "Bill Hagerty",
    },

    # --- SENATE ARMED SERVICES (119th) ---
    # Source: armed-services.senate.gov, senate.gov assignments
    (119, "Senate", "Armed Services"): {
        "Dan Sullivan", "Tommy Tuberville", "Markwayne Mullin",
        "Rick Scott", "Jim Banks", "Ashley Moody",
        # Democrats
        "Richard Blumenthal", "Tammy Duckworth",
        "Angus King", "Mark Warner",
    },

    # --- SENATE COMMERCE, SCIENCE & TRANSPORTATION (119th) ---
    # Source: commerce.senate.gov, senate.gov assignments
    (119, "Senate", "Commerce"): {
        "Ted Cruz", "Cynthia Lummis", "Dan Sullivan",
        "Jerry Moran", "Shelley Moore Capito",
        "John Neely Kennedy", "Rick Scott",
        "John Curtis",
        # Democrats
        "Tammy Duckworth", "Gary Peters",
        "Richard Blumenthal", "Tina Smith",
        "John Fetterman", "John Hickenlooper",
    },
}


def get_congress_number(date_str: str) -> int:
    """Determine Congress number from a date string.
    118th: Jan 3, 2023 – Jan 3, 2025
    119th: Jan 3, 2025 – Jan 3, 2027
    """
    # Capitol Trades uses "Sept" instead of standard "Sep"
    date_str = date_str.strip().replace("Sept", "Sep")
    
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.year < 2025:
                return 118
            elif dt.year == 2025 and dt.month == 1 and dt.day < 3:
                return 118
            else:
                return 119
        except ValueError:
            continue
    # Fallback: if date contains relative terms or year keywords
    if "2023" in date_str or "2024" in date_str:
        return 118
    return 119


def build_lookup():
    """Build a flat lookup: (congress, politician_name) -> list of committee names."""
    lookup = {}
    for (congress, chamber, committee), members in POWER_COMMITTEES.items():
        for name in members:
            key = (congress, name)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(committee)
    return lookup


def is_power_committee(name: str, congress: int, lookup: dict) -> tuple:
    """Returns (is_power: bool, committees: list[str])."""
    key = (congress, name)
    if key in lookup:
        return True, lookup[key]
    return False, []


def augment_csv(input_path: str, output_path: str):
    """Read capitol_trades.csv, add PowerCommittee column, write output."""
    lookup = build_lookup()
    matched = 0
    total = 0

    with open(input_path, "r", encoding="utf-8-sig") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["power_committee", "power_committee_names"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            name = row["politician_name"].strip()
            traded_date = row["traded_date"].strip()
            congress = get_congress_number(traded_date)

            is_power, committees = is_power_committee(name, congress, lookup)
            row["power_committee"] = 1 if is_power else 0
            row["power_committee_names"] = "|".join(committees) if committees else ""
            writer.writerow(row)

            if is_power:
                matched += 1

    print(f"Processed {total} trades.")
    print(f"  Power committee trades: {matched} ({matched/total*100:.1f}%)")
    print(f"  Non-power committee trades: {total - matched} ({(total-matched)/total*100:.1f}%)")
    print(f"Output written to: {output_path}")


def audit(input_path: str):
    """Print audit report: which politicians are/aren't matched."""
    lookup = build_lookup()

    # Collect unique (name, chamber) pairs and a sample trade date
    politicians = {}
    with open(input_path, "r", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            name = row["politician_name"].strip()
            chamber = row.get("chamber", "").strip()
            traded_date = row["traded_date"].strip()
            if name not in politicians:
                politicians[name] = {"chamber": chamber, "date": traded_date}

    print(f"\n{'='*70}")
    print(f"AUDIT REPORT: Power Committee Classification")
    print(f"{'='*70}\n")

    power_members = []
    non_power_members = []

    for name, info in sorted(politicians.items()):
        congress = get_congress_number(info["date"])
        is_power, committees = is_power_committee(name, congress, lookup)
        # Also check other congress
        other_congress = 119 if congress == 118 else 118
        is_power_other, committees_other = is_power_committee(name, other_congress, lookup)

        if is_power or is_power_other:
            all_comms = set(committees + committees_other)
            power_members.append((name, info["chamber"], all_comms))
        else:
            non_power_members.append((name, info["chamber"]))

    print(f"MATCHED to power committee ({len(power_members)} legislators):")
    print(f"{'-'*70}")
    for name, chamber, comms in power_members:
        print(f"  {name:<35} {chamber:<8} {', '.join(sorted(comms))}")

    print(f"\nNOT matched ({len(non_power_members)} legislators):")
    print(f"{'-'*70}")
    for name, chamber in non_power_members:
        print(f"  {name:<35} {chamber}")

    print(f"\n{'='*70}")
    print("VERIFY: Check unmatched legislators against official rosters.")
    print("  House: appropriations.house.gov, waysandmeans.house.gov,")
    print("         armedservices.house.gov, energycommerce.house.gov")
    print("  Senate: senate.gov/general/committee_assignments/assignments.htm")
    print(f"{'='*70}\n")


def verify_name(input_path: str, name: str):
    """Check a specific legislator."""
    lookup = build_lookup()
    for congress in [118, 119]:
        is_power, committees = is_power_committee(name, congress, lookup)
        status = "YES" if is_power else "NO"
        comms = ", ".join(committees) if committees else "(none)"
        print(f"  {congress}th Congress: {status} — {comms}")


if __name__ == "__main__":
    input_path = "/mnt/user-data/uploads/capitol_trades.csv"
    output_path = "/mnt/user-data/outputs/capitol_trades_augmented.csv"

    if "--audit" in sys.argv:
        audit(input_path)
    elif "--verify" in sys.argv:
        idx = sys.argv.index("--verify")
        name = " ".join(sys.argv[idx + 1:])
        print(f"Verifying: {name}")
        verify_name(input_path, name)
    else:
        augment_csv(input_path, output_path)
