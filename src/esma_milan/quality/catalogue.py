"""The catalogue of the data quality report: every field and every rule.

Mirrors nothing in the R reference. The rules are specified in
docs/data-quality/specification.md (sections 3.2, 4 and 6); the per-field
sources are in docs/data-quality/rre-source-mapping.csv.

The catalogue is reviewed data, written out here and not derived at run
time. Tests keep it equal to what it was transcribed from:

  field names, formats, ND flags   the bundled taxonomy and the mapping file
  code lists                       the codes in the taxonomy's CONTENT TO
                                   REPORT text
  the rules applied to each field  the mapping file's implementing_checks
                                   column, in both directions

`FIELDS` holds the 107 template fields and the two optional XSD elements.
`RULES` holds every rule of the specification, including the ones a later
release implements, so a report can say what it did not assess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from esma_milan.config import (
    DQ_CRITICAL_FIELDS,
    DQ_PERCENT_SCALE_EXCLUDED_FIELDS,
    DQ_POSSIBLE_PLACEHOLDERS,
)

FileKind = Literal["loans", "collateral"]
FormatKind = Literal[
    "MONETARY", "PERCENTAGE", "INTEGER", "DATE", "YEAR", "BOOLEAN", "LIST",
    "NUTS", "COUNTRY", "CURRENCY", "LEI", "ALPHANUM",
]
Basis = Literal["E", "L", "X", "A", "P"]
Severity = Literal["Stop", "Warn", "Info"]
ResultKind = Literal[
    "violation", "accepted alternative", "context", "question", "limitation"
]
RuleStatus = Literal["implemented", "partly_implemented", "not_implemented"]

RESULT_KINDS: tuple[str, ...] = (
    "violation", "accepted alternative", "context", "question", "limitation"
)

_FORMAT_KINDS: dict[str, FormatKind] = {
    "MONETARY": "MONETARY", "PERCENTAGE": "PERCENTAGE", "INTEGER-9999": "INTEGER",
    "DATEFORMAT": "DATE", "YEAR": "YEAR", "Y/N": "BOOLEAN", "LIST": "LIST",
    "NUTS": "NUTS", "COUNTRYCODE_2": "COUNTRY", "CURRENCYCODE_3": "CURRENCY",
    "LEI": "LEI",
}

# Kinds in which no valid value can start with "ND4" (specification 6.1,
# classification step 2).
ND4_PREFIX_KINDS: frozenset[str] = frozenset({
    "MONETARY", "PERCENTAGE", "INTEGER", "DATE", "YEAR", "BOOLEAN", "LIST",
    "CURRENCY", "COUNTRY", "NUTS",
})
# Kinds whose XSD type collapses surrounding XML whitespace (P4(a)).
WHITESPACE_COLLAPSING_KINDS: frozenset[str] = frozenset({
    "MONETARY", "PERCENTAGE", "INTEGER", "DATE", "YEAR", "BOOLEAN",
})
NUMERIC_KINDS: frozenset[str] = frozenset({"MONETARY", "PERCENTAGE", "INTEGER"})

# P6(a): "the numerical field must be equal to or greater than 0"
# (validation workbook 1.3.1; specification 4.2 item 4).
_NOT_BELOW_ZERO: frozenset[str] = frozenset({
    "RREL16", "RREL20", "RREL29", "RREL31", "RREL32", "RREL33", "RREL41", "RREL61",
    "RREL64", "RREL67", "RREL71", "RREL77", "RREC13", "RREC17", "RREC21",
})
# P6(b): "must be equal to or less than the Data Cut-off Date" (item 5).
_NOT_AFTER_CUTOFF: frozenset[str] = frozenset({
    "RREL7", "RREL8", "RREL9", "RREL23", "RREL63", "RREL65", "RREL66", "RREL72",
    "RREC15", "RREC19", "RREC20",
})
# Currency evidence case 2: the dedicated field that states an amount's
# currency when its companion column does not (specification 6.2).
_DEDICATED_CURRENCY: dict[str, str] = {
    "RREL29": "RREL28", "RREL30": "RREL28", "RREL16": "RREL18", "RREC13": "RREC22",
}
# Fields whose values reach the MILAN output (specification Appendix B).
_FEEDS_MILAN: frozenset[str] = frozenset(
    [f"RREL{n}" for n in (
        2, 3, 4, 5, 8, 9, 10, 11, 13, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30,
        31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 50, 51, 58, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 82,
    )]
    + [f"RREC{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23)]
)
# Collateral fields whose ND5 is tied to the collateral type (P3f).
CONDITIONAL_ND5_FIELDS: frozenset[str] = frozenset({"RREC7", "RREC8", "RREC9", "RREC12", "RREC16"})
PROPERTY_COLLATERAL_TYPES: frozenset[str] = frozenset({"CBLD", "RBLD", "IBLD", "OTRE"})
UNDETERMINED_COLLATERAL_TYPES: frozenset[str] = frozenset({"OTHR", "MIXD"})


@dataclass(frozen=True)
class FieldEntry:
    """One template field, or one of the two optional XSD elements."""

    code: str
    """ESMA field code; `OPT_RENT_CURRENT` / `OPT_RENT_ORIGINAL` for the
    optional elements, which have no RTS code."""
    file: FileKind
    name: str
    taxonomy_format: str
    """As the mapping file writes it, e.g. "ALPHANUM-1000"."""
    kind: FormatKind
    max_length: int | None
    nd1_4_allowed: bool
    nd5_allowed: bool
    codes: frozenset[str] | None
    optional: bool
    critical: bool
    feeds_milan: bool
    not_below_zero: bool
    not_after_cutoff: bool
    dedicated_currency: str | None
    final_report_category: int | None
    """Provisional category of specification 3.3 (used by L1, a later release)."""
    sources: str

    @property
    def label(self) -> str:
        return f"{self.name} (optional element)" if self.optional else f"{self.code} {self.name}"


@dataclass(frozen=True)
class RuleEntry:
    """One rule of the specification (3.2)."""

    rule_id: str
    title: str
    basis: Basis
    regulatory_source: str | None
    """Set when a violation of the rule enters the compliance judgment."""
    severity: Severity
    applies_to: str
    counts: bool
    """Whether a violation on a critical field counts towards the
    acceptance rules of a later release."""
    defect_key: str
    """The underlying defect; `{field}` is the field code. Two rules with
    one key on one field measure one defect."""
    needs: tuple[str, ...]
    result_kinds: tuple[ResultKind, ...]
    status: RuleStatus
    not_implemented_part: str | None = None


_FIELD_ROWS: tuple[tuple[str, str, str, bool, bool, int | None, str], ...] = (
    ("RREL1", "Unique Identifier", "ALPHANUM-28", False, False, 1, "SecuritisationIdentifier; workbook rows 9;59702"),
    ("RREL2", "Original Underlying Exposure Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 14"),
    ("RREL3", "New Underlying Exposure Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 13"),
    ("RREL4", "Original Obligor Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 16"),
    ("RREL5", "New Obligor Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 15"),
    ("RREL6", "Data Cut-Off Date", "DATEFORMAT", False, False, 1, "ISODate; workbook rows 10;59705"),
    ("RREL7", "Pool Addition Date", "DATEFORMAT", False, True, 3, "ISODate; workbook rows 23;24;26"),
    ("RREL8", "Date Of Repurchase", "DATEFORMAT", False, True, 1, "ISODate; workbook rows 31;32;34"),
    ("RREL9", "Redemption Date", "DATEFORMAT", False, True, 1, "ISODate; workbook rows 988;989;991"),
    ("RREL10", "Resident", "Y/N", True, False, 4, "YesNoIndicator; workbook rows 41;42;44;50;51"),
    ("RREL11", "Geographic Region - Obligor", "NUTS", True, False, 4, "TerritorialUnitDomainIdentifier; workbook rows 58;59;61;67;68"),
    ("RREL12", "Geographic Region Classification", "YEAR", True, False, 4, "ISOYear; workbook rows 74;75;77;83;84"),
    ("RREL13", "Employment Status", "LIST", True, False, 4, "EmploymentStatus1Code; workbook rows 241;242;254;260;261"),
    ("RREL14", "Credit Impaired Obligor", "Y/N", False, True, 3, "YesNoIndicator; workbook rows 267;268;270"),
    ("RREL15", "Customer Type", "LIST", True, False, 4, "CustomerType1Code; workbook rows 91;92;102;108;109"),
    ("RREL16", "Primary Income", "MONETARY", True, False, 4, "ActiveOrHistoricCurrencyAndAmount; workbook rows 117;119;120;123;126;132;133"),
    ("RREL17", "Primary Income Type", "LIST", True, False, 4, "PrimaryIncomeType1Code; workbook rows 139;140;153;159;160"),
    ("RREL18", "Primary Income Currency", "CURRENCYCODE_3", True, False, 4, "ActiveOrHistoricCurrencyCode (attribute); workbook rows "),
    ("RREL19", "Primary Income Verification", "LIST", True, False, 4, "IncomeVerification1Code; workbook rows 166;167;176;182;183"),
    ("RREL20", "Secondary Income", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 191;193;194;197;200;207;208"),
    ("RREL21", "Secondary Income Verification", "LIST", True, True, 3, "IncomeVerification1Code; workbook rows 214;215;224;231;232"),
    ("RREL22", "Special Scheme", "ALPHANUM-10000", True, True, 3, "Max10000Text; workbook rows 278;279;281;288;289"),
    ("RREL23", "Origination Date", "DATEFORMAT", True, False, 4, "ISODate; workbook rows 351;352;354;360;361"),
    ("RREL24", "Maturity Date", "DATEFORMAT", False, True, 3, "ISODate; workbook rows 367;368;370"),
    ("RREL25", "Original Term", "INTEGER-9999", True, True, 3, "Max4PositiveNumber; workbook rows 375;376;378;385;386"),
    ("RREL26", "Origination Channel", "LIST", True, True, 3, "OriginationChannel1Code; workbook rows 295;296;306;313;314"),
    ("RREL27", "Purpose", "LIST", True, False, 4, "LoanPurpose1Code; workbook rows 320;321;337;343;344"),
    ("RREL28", "Currency Denomination", "CURRENCYCODE_3", False, False, 4, "ActiveOrHistoricCurrencyCode (attribute); workbook rows "),
    ("RREL29", "Original Principal Balance", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 416;418;419;422;425;432;433"),
    ("RREL30", "Current Principal Balance", "MONETARY", False, True, 2, "ActiveOrHistoricCurrencyAndAmount; workbook rows 394;396;397;400;403"),
    ("RREL31", "Prior Principal Balances", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 439;441;442;445;448;455;456"),
    ("RREL32", "Pari Passu Underlying Exposures", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 462;464;465;468;471;478;479"),
    ("RREL33", "Total Credit Limit", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 485;487;488;491;494"),
    ("RREL34", "Purchase Price", "PERCENTAGE", False, True, 3, "PercentageRate; workbook rows 408;409;411"),
    ("RREL35", "Amortisation Type", "LIST", True, False, 4, "AmortisationType1Code; workbook rows 539;540;548;554;555"),
    ("RREL36", "Principal Grace Period End Date", "DATEFORMAT", False, True, 3, "ISODate; workbook rows 502;503;505"),
    ("RREL37", "Scheduled Principal Payment Frequency", "LIST", False, True, 3, "EventFrequency11Code; workbook rows 510;511;519"),
    ("RREL38", "Scheduled Interest Payment Frequency", "LIST", False, True, 3, "EventFrequency11Code; workbook rows 524;525;533"),
    ("RREL39", "Payment Due", "MONETARY", False, True, 2, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1031;1033;1034;1037;1040"),
    ("RREL40", "Debt To Income Ratio", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 934;935;937;944;945"),
    ("RREL41", "Balloon Amount", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 561;563;564;567;570;577;578"),
    ("RREL42", "Interest Rate Type", "LIST", False, True, 3, "InterestRateType2Code; workbook rows 657;658;674"),
    ("RREL43", "Current Interest Rate", "PERCENTAGE", False, True, 2, "PercentageRate; workbook rows 587;588;590"),
    ("RREL44", "Current Interest Rate Index", "LIST", False, True, 3, "InterestRateIndex1Code; workbook rows 595;596;629"),
    ("RREL45", "Current Interest Rate Index Tenor", "LIST", False, True, 3, "InterestRateIndexTenor1Code; workbook rows 634;635;651"),
    ("RREL46", "Current Interest Rate Margin", "PERCENTAGE", False, True, 2, "PercentageRate; workbook rows 679;680;682"),
    ("RREL47", "Interest Rate Reset Interval", "INTEGER-9999", False, True, 3, "Max4PositiveNumber; workbook rows 688;689;691"),
    ("RREL48", "Interest Rate Cap", "PERCENTAGE", False, True, 3, "PercentageRate; workbook rows 696;697;699"),
    ("RREL49", "Interest Rate Floor", "PERCENTAGE", False, True, 3, "PercentageRate; workbook rows 704;705;707"),
    ("RREL50", "Revision Margin 1", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 714;715;717;724;725"),
    ("RREL51", "Interest Revision Date 1", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 731;732;734;741;742"),
    ("RREL52", "Revision Margin 2", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 750;751;753;760;761"),
    ("RREL53", "Interest Revision Date 2", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 767;768;770;777;778"),
    ("RREL54", "Revision Margin 3", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 786;787;789;796;797"),
    ("RREL55", "Interest Revision Date 3", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 803;804;806;813;814"),
    ("RREL56", "Revised Interest Rate Index", "LIST", True, True, 3, "InterestRateIndex1Code; workbook rows 821;822;855;862;863"),
    ("RREL57", "Revised Interest Rate Index Tenor", "LIST", True, True, 3, "InterestRateIndexTenor1Code; workbook rows 869;870;886;893;894"),
    ("RREL58", "Number Of Payments Before Securitisation", "INTEGER-9999", True, False, 4, "Max4PositiveNumber; workbook rows 1015;1016;1018;1024;1025"),
    ("RREL59", "Percentage Of Prepayments Allowed Per Year", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 1160;1161;1163;1170;1171"),
    ("RREL60", "Prepayment Lock-Out End Date", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1249;1250;1252;1259;1260"),
    ("RREL61", "Prepayment Fee", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1177;1179;1180;1183;1186"),
    ("RREL62", "Prepayment Fee End Date", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1191;1192;1194;1201;1202"),
    ("RREL63", "Prepayment Date", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1208;1209;1211;1218;1219"),
    ("RREL64", "Cumulative Prepayments", "MONETARY", True, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1225;1227;1228;1231;1234;1241;1242"),
    ("RREL65", "Date Of Restructuring", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 996;997;999;1006;1007"),
    ("RREL66", "Date Last In Arrears", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1047;1048;1050;1057;1058"),
    ("RREL67", "Arrears Balance", "MONETARY", False, False, 2, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1065;1066;1069"),
    ("RREL68", "Number Of Days In Arrears", "INTEGER-9999", False, False, 2, "Max4PositiveNumber; workbook rows 1071"),
    ("RREL69", "Account Status", "LIST", False, False, 1, "SecuritisationAccountStatus1Code; workbook rows 971"),
    ("RREL70", "Reason for Default or Foreclosure", "LIST", True, True, 3, "ReasonForDefaultOrForclosure1Code; workbook rows 1083;1084;1090;1097;1098"),
    ("RREL71", "Default Amount", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1104;1106;1107;1110;1113"),
    ("RREL72", "Default Date", "DATEFORMAT", False, True, 3, "ISODate; workbook rows 1075;1076;1078"),
    ("RREL73", "Allocated Losses", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1142;1144;1145;1148;1151"),
    ("RREL74", "Cumulative Recoveries", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1128;1130;1131;1134;1137"),
    ("RREL75", "Litigation", "Y/N", False, True, 3, "YesNoIndicator; workbook rows 1119;1120;1122"),
    ("RREL76", "Recourse", "Y/N", True, True, 3, "YesNoIndicator; workbook rows 903;904;906;913;914"),
    ("RREL77", "Deposit Amount", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 920;922;923;926;929"),
    ("RREL78", "Insurance Or Investment Provider", "ALPHANUM-1000", True, True, 3, "Max1000Text; workbook rows 951;952;954;961;962"),
    ("RREL79", "Original Lender Name", "ALPHANUM-100", True, True, 3, "Max100Text; workbook rows 1277;1278;1280;1287;1288"),
    ("RREL80", "Original Lender Legal Entity Identifier", "LEI", True, True, 3, "LEIIdentifier; workbook rows 1294;1295;1297;1304;1305"),
    ("RREL81", "Original Lender Establishment Country", "COUNTRYCODE_2", True, True, 3, "CountryCode; workbook rows 1311;1312;1316;1323;1324"),
    ("RREL82", "Originator Name", "ALPHANUM-100", False, False, 4, "Max100Text; workbook rows 1270"),
    ("RREL83", "Originator Legal Entity Identifier", "LEI", False, False, 4, "LEIIdentifier; workbook rows 1271"),
    ("RREL84", "Originator Establishment Country", "COUNTRYCODE_2", False, False, 4, "CountryCode; workbook rows 1272"),
    ("RREC1", "Unique Identifier", "ALPHANUM-28", False, False, 1, "SecuritisationIdentifier; workbook rows 59701;9;59702;59704"),
    ("RREC2", "Underlying Exposure Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 13"),
    ("RREC3", "Original Collateral Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 1335"),
    ("RREC4", "New Collateral Identifier", "ALPHANUM-1000", False, False, 1, "Max1000Text; workbook rows 1336"),
    ("RREC5", "Collateral Type", "LIST", False, False, 4, "CollateralType7Code; workbook rows 1358"),
    ("RREC6", "Geographic Region - Collateral", "NUTS", True, True, 3, "TerritorialUnitDomainIdentifier; workbook rows 1341;1342;1344;1351;1352"),
    ("RREC7", "Occupancy Type", "LIST", True, True, 3, "OccupancyType1Code; workbook rows 1430;1431;1439;1446;1447"),
    ("RREC8", "Lien", "INTEGER-9999", True, True, 3, "Max4PositiveNumber; workbook rows 1385;1386;1388;1395;1396"),
    ("RREC9", "Property Type", "LIST", False, True, 3, "PropertyType1Code; workbook rows 1453;1454;1466"),
    ("RREC10", "Energy Performance Certificate Value", "LIST", True, True, 3, "EnergyPerformanceCertificateClass1Code; workbook rows 1471;1472;1483;1490;1491"),
    ("RREC11", "Energy Performance Certificate Provider Name", "ALPHANUM-100", True, True, 3, "Max100Text; workbook rows 1497;1498;1500;1507;1508"),
    ("RREC12", "Current Loan-To-Value", "PERCENTAGE", True, True, 2, "PercentageRate; workbook rows 1600;1601;1603;1610;1611"),
    ("RREC13", "Current Valuation Amount", "MONETARY", True, True, 2, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1617;1619;1620;1623;1626;1633;1634"),
    ("RREC14", "Current Valuation Method", "LIST", True, False, 4, "ValuationMethod1Code; workbook rows 1640;1641;1653;1659;1660"),
    ("RREC15", "Current Valuation Date", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1666;1667;1669;1676;1677"),
    ("RREC16", "Original Loan-To-Value", "PERCENTAGE", True, True, 3, "PercentageRate; workbook rows 1517;1518;1520;1527;1528"),
    ("RREC17", "Original Valuation Amount", "MONETARY", True, False, 4, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1534;1536;1537;1540;1543;1549;1550"),
    ("RREC18", "Original Valuation Method", "LIST", True, False, 4, "ValuationMethod1Code; workbook rows 1556;1557;1569;1575;1576"),
    ("RREC19", "Original Valuation Date", "DATEFORMAT", True, False, 4, "ISODate; workbook rows 1582;1583;1585;1591;1592"),
    ("RREC20", "Date Of Sale", "DATEFORMAT", True, True, 3, "ISODate; workbook rows 1686;1687;1689;1696;1697"),
    ("RREC21", "Sale Price", "MONETARY", False, True, 3, "ActiveOrHistoricCurrencyAndAmount; workbook rows 1703;1705;1706;1709;1712"),
    ("RREC22", "Collateral Currency", "CURRENCYCODE_3", False, True, 3, "ActiveOrHistoricCurrencyCode (attribute); workbook rows "),
    ("RREC23", "Guarantor Type", "LIST", True, False, 4, "GuarantorType1Code; workbook rows 1403;1404;1417;1423;1424"),
)

CODE_LISTS: dict[str, frozenset[str]] = {
    "RREL13": frozenset({
        "EMBL", "EMRS", "EMUK", "NOEM", "OTHR", "PNNR", "SFEM", "STNT", "UNEM",
    }),
    "RREL15": frozenset({
        "CEMO", "CNEO", "CNRO", "EEMO", "ENEO", "ENRO", "OTHR",
    }),
    "RREL17": frozenset({
        "CORP", "DSPL", "EISS", "EITX", "ENIS", "GRAN", "NITS", "NITX", "NTIN", "OTHR",
    }),
    "RREL19": frozenset({
        "NVRF", "OTHR", "SCNF", "SCRG", "SCRT", "VRFD",
    }),
    "RREL21": frozenset({
        "NVRF", "OTHR", "SCNF", "SCRG", "SCRT", "VRFD",
    }),
    "RREL26": frozenset({
        "BRAN", "BROK", "DRCT", "OTHR", "TPAC", "TPTC", "WEBI",
    }),
    "RREL27": frozenset({
        "BSFN", "CMRT", "CNST", "DCON", "EQRE", "GSPL", "IMRT", "OTHR", "PURC", "RENV",
        "RGBY", "RMEQ", "RMRT",
    }),
    "RREL35": frozenset({
        "BLLT", "DEXX", "FIXE", "FRXX", "OTHR",
    }),
    "RREL37": frozenset({
        "MNTH", "OTHR", "QUTR", "SEMI", "YEAR",
    }),
    "RREL38": frozenset({
        "MNTH", "OTHR", "QUTR", "SEMI", "YEAR",
    }),
    "RREL42": frozenset({
        "CAPP", "DISC", "FINX", "FLCA", "FLCF", "FLFL", "FLIF", "FXPR", "FXRL", "MODE",
        "OBLS", "OTHR", "SWIC",
    }),
    "RREL44": frozenset({
        "BBSW", "BOER", "BUBO", "CDOR", "CIBO", "ECBR", "EONA", "EONS", "EUCH", "EURI",
        "EUUS", "FUSW", "GCFR", "ISDA", "JIBA", "LDOR", "LIBI", "LIBO", "MAAA", "MOSP",
        "NIBO", "OTHR", "PFAN", "PRBO", "STBO", "SWAP", "TIBO", "TLBO", "TREA", "WIBO",
    }),
    "RREL45": frozenset({
        "DAIL", "FOMN", "INDA", "MNTH", "ONDE", "OTHR", "OVNG", "QUTR", "SEMI", "TOMN",
        "TOWK", "WEEK", "YEAR",
    }),
    "RREL56": frozenset({
        "BBSW", "BOER", "BUBO", "CDOR", "CIBO", "ECBR", "EONA", "EONS", "EUCH", "EURI",
        "EUUS", "FUSW", "GCFR", "ISDA", "JIBA", "LDOR", "LIBI", "LIBO", "MAAA", "MOSP",
        "NIBO", "OTHR", "PFAN", "PRBO", "STBO", "SWAP", "TIBO", "TLBO", "TREA", "WIBO",
    }),
    "RREL57": frozenset({
        "DAIL", "FOMN", "INDA", "MNTH", "ONDE", "OTHR", "OVNG", "QUTR", "SEMI", "TOMN",
        "TOWK", "WEEK", "YEAR",
    }),
    "RREL69": frozenset({
        "ARRE", "DADB", "DFLT", "DTCR", "NDFT", "OTHR", "PERF", "RARR", "RDMD", "REBR",
        "REDF", "REOT", "RERE", "RESS", "RNAR",
    }),
    "RREL70": frozenset({
        "PDXX", "UPPD", "UPXX",
    }),
    "RREC5": frozenset({
        "AERO", "CARX", "CBLD", "CMTR", "ENEQ", "GUAR", "IBLD", "INDE", "INDV", "ITEQ",
        "MCHT", "MDEQ", "MIXD", "NACM", "NALV", "OFEQ", "OTFA", "OTGI", "OTHE", "OTHR",
        "OTHV", "OTRE", "RALV", "RBLD", "SECU",
    }),
    "RREC7": frozenset({
        "FOWN", "HOLD", "OTHR", "POWN", "TLET",
    }),
    "RREC9": frozenset({
        "BIZZ", "LAND", "MULF", "OTHR", "PCMM", "RBGL", "RFLT", "RHOS", "RTHS",
    }),
    "RREC10": frozenset({
        "EPCA", "EPCB", "EPCC", "EPCD", "EPCE", "EPCF", "EPCG", "OTHR",
    }),
    "RREC14": frozenset({
        "AUVM", "DKTP", "DRVB", "FIEI", "FOEI", "IDXD", "MAEA", "OTHR", "TXAT",
    }),
    "RREC18": frozenset({
        "AUVM", "DKTP", "DRVB", "FIEI", "FOEI", "IDXD", "MAEA", "OTHR", "TXAT",
    }),
    "RREC23": frozenset({
        "BANK", "CATN", "FAML", "FGAS", "GOVE", "INSU", "IOTH", "NGUA", "NHGX", "OTHR",
    }),
}


def _entry(row: tuple[str, str, str, bool, bool, int | None, str]) -> FieldEntry:
    code, name, fmt, nd14, nd5, category, sources = row
    alphanum = fmt.startswith("ALPHANUM-")
    return FieldEntry(
        code=code,
        file="loans" if code.startswith("RREL") else "collateral",
        name=name,
        taxonomy_format=fmt,
        kind="ALPHANUM" if alphanum else _FORMAT_KINDS[fmt],
        max_length=int(fmt.split("-")[1]) if alphanum else None,
        nd1_4_allowed=nd14,
        nd5_allowed=nd5,
        codes=CODE_LISTS.get(code),
        optional=False,
        critical=code in DQ_CRITICAL_FIELDS,
        feeds_milan=code in _FEEDS_MILAN,
        not_below_zero=code in _NOT_BELOW_ZERO,
        not_after_cutoff=code in _NOT_AFTER_CUTOFF,
        dedicated_currency=_DEDICATED_CURRENCY.get(code),
        final_report_category=category,
        sources=sources,
    )


def _optional(code: str, name: str) -> FieldEntry:
    # Specification 4.2 item 17 and 6.4 (O1): an amount with currency, not
    # below 0, or ND5; ND1 to ND4 are not offered.
    return FieldEntry(
        code=code, file="collateral", name=name, taxonomy_format="MONETARY",
        kind="MONETARY", max_length=None, nd1_4_allowed=False, nd5_allowed=True,
        codes=None, optional=True, critical=False, feeds_milan=False,
        not_below_zero=True, not_after_cutoff=False, dedicated_currency=None,
        final_report_category=None,
        sources="Monetary2Choice (GrssAnlRntlIncm); XSD 1.3.0, amendment 78",
    )


FIELDS: dict[str, FieldEntry] = {
    **{row[0]: _entry(row) for row in _FIELD_ROWS},
    "OPT_RENT_CURRENT": _optional("OPT_RENT_CURRENT", "Current gross annual rental income"),
    "OPT_RENT_ORIGINAL": _optional("OPT_RENT_ORIGINAL", "Original gross annual rental income"),
}


def fields_of(file: FileKind, *, optional: bool = False) -> tuple[FieldEntry, ...]:
    """The template fields of one file, in template order; with
    `optional=True`, the optional elements instead."""
    return tuple(f for f in FIELDS.values() if f.file == file and f.optional is optional)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_WORKBOOK = "ESMA validation workbook 1.3.1"
_RTS = "Delegated Regulation (EU) 2020/1224"
_ITS = "Implementing Regulation (EU) 2020/1225, Annex I"


def _rule(
    rule_id: str, title: str, basis: Basis, severity: Severity, defect_key: str,
    kinds: tuple[ResultKind, ...], *, source: str | None = None, counts: bool = False,
    applies_to: str = "every record of the field's file",
    needs: tuple[str, ...] = (), status: RuleStatus = "implemented",
    missing: str | None = None,
) -> RuleEntry:
    return RuleEntry(
        rule_id=rule_id, title=title, basis=basis, regulatory_source=source,
        severity=severity, applies_to=applies_to, counts=counts, defect_key=defect_key,
        needs=needs, result_kinds=kinds, status=status, not_implemented_part=missing,
    )


_V: tuple[ResultKind, ...] = ("violation",)
_P_RULES: tuple[RuleEntry, ...] = (
    _rule("P1", "every template field is reported (the field has a column in the file)",
          "E", "Warn", "field_absent:{field}", _V, counts=True,
          source=f"{_RTS}, Annex 2; Q&A 1283", applies_to="each of the 107 template fields"),
    _rule("P2", "value-state profile of the field", "P", "Info", "profile:{field}", ("context",)),
    _rule("P3a", "ND1 to ND4 used only where Annex 2 allows them", "E", "Warn",
          "nd_not_allowed:{field}", _V, counts=True, source=f"{_RTS}, Article 9(3) and Annex 2"),
    _rule("P3b", "ND5 used only where Annex 2 allows it", "E", "Warn",
          "nd_not_allowed:{field}", _V, counts=True, source=f"{_RTS}, Article 9(3) and Annex 2"),
    _rule("P3c", "no empty cell: a value or a permitted ND option is reported", "E", "Warn",
          "value_missing:{field}", _V, counts=True, source=f"{_RTS}, Article 9(3); Q&A 1283"),
    _rule("P3d", "no missing-value marker other than the ND options", "E", "Warn",
          "value_missing:{field}", _V, counts=True, source=f"{_RTS}, Article 9(3); Q&A 1290"),
    _rule("P3e", "ND4 is written ND4-YYYY-MM-DD with a real date after the data cut-off date",
          "E", "Warn", "nd4_date:{field}", _V, counts=True, source=f"{_RTS}, Article 9(3)",
          needs=("the data cut-off date of the record, for the date comparison",)),
    _rule("P3f", "ND5 is consistent with the applicability the record itself establishes",
          "E", "Warn", "conditional_nd5:{field}", ("violation", "question", "limitation"),
          source=f"{_RTS}, Annex 2 field definitions; Q&A 1296, 1323(c)",
          applies_to="collateral records, for RREC7, RREC8, RREC9, RREC12 and RREC16; "
                     "RREL65 cells that mix a date with an ND option",
          needs=("RREC5 on the same record",), status="partly_implemented",
          missing="ND5 on a property record depends on the lifecycle of the exposure, and "
                  "RREC12 = 0 on a negative balance needs the loan; both need the link "
                  "between collateral and loan records (families E and F)"),
    _rule("P3g", "what the pipeline's loader does with the value", "L", "Warn",
          "loader_effect:{field}", ("violation", "context"),
          applies_to="cells the loader reads differently from what the file states"),
    _rule("P4(a)", "surrounding XML whitespace around a number, boolean, date or year",
          "A", "Info", "padding:{field}", ("accepted alternative",),
          applies_to="values of the types whose XSD type collapses whitespace"),
    _rule("P4(c)", "no control character or line break inside a typed value", "E", "Warn",
          "format:{field}", _V, source=_ITS, status="implemented",
          missing="reported by P5 with the reason 'control_character' (one defect, one row)"),
    _rule("P4(d)", "a code the pipeline compares exactly is written exactly", "L", "Warn",
          "format:{field}", _V, counts=True,
          applies_to="code-list fields the pipeline reads, where the value is a list code "
                     "only after trimming or upper-casing"),
    _rule("P5", "the value has the format prescribed for the field", "E", "Warn",
          "format:{field}", ("violation", "accepted alternative"), counts=True,
          source=f"{_ITS}; ESMA disclosure XSD 1.3.0",
          applies_to="present values of every field except {ALPHANUM-n}"),
    _rule("P6(a)", "the amount is not below 0", "E", "Warn", "amount_below_zero:{field}", _V,
          source=f"{_WORKBOOK}: the numerical field must be equal to or greater than 0",
          needs=("the value parses as a number",)),
    _rule("P6(b)", "the date is not later than the data cut-off date", "X", "Warn",
          "date_after_cutoff:{field}", _V,
          source=f"{_WORKBOOK}: must be equal to or less than the Data Cut-off Date",
          needs=("the value parses as a date", "the data cut-off date of the record")),
    _rule("P6(c)", "RREL59 is between 0 and 100 inclusive", "E", "Warn",
          "out_of_range:{field}", _V, source=f"{_WORKBOOK}: between 0 and 100",
          needs=("the value parses as a number",)),
    _rule("P6(d)", "RREL12 is greater than 1980 and less than 2100", "E", "Warn",
          "out_of_range:{field}", _V, source=f"{_WORKBOOK}: greater than 1980, less than 2100",
          needs=("the value parses as a year",)),
    _rule("P6(e)", "RREL6 is in the past", "E", "Warn", "cutoff_not_in_past", _V,
          source=f"{_WORKBOOK}: must be in the past", status="not_implemented",
          missing="owned and reported once by A10(a) (family A)"),
    _rule("P7(a)", "the text is not longer than the field allows", "E", "Warn",
          "text_length:{field}", _V, counts=True, source=f"{_ITS}; XSD length facet"),
    _rule("P7(b)", "the text is ASCII without lower-case letters", "E", "Warn",
          "text_characters:{field}", _V,
          source=f"{_ITS} (the XSD restricts the length only and accepts lower case)"),
    _rule("P7(c)", "the unique identifier is the reporting entity's LEI, N, a four-digit "
          "year and a two-digit number", "E", "Warn", "identifier_construction:{field}", _V,
          source=f"{_RTS}, Article 11(1); XSD pattern; ISO 17442 for the LEI check digits",
          applies_to="RREL1 and RREC1"),
    _rule("P8", "possible placeholder", "P", "Warn", "placeholder:{field}", ("question",),
          applies_to="the fields and values of DQ_POSSIBLE_PLACEHOLDERS"),
    _rule("P9", "cardinality: a column with one value on every record", "P", "Info",
          "cardinality:{field}", ("context",)),
    _rule("P10", "MILAN lineage of the field's values", "L", "Info", "lineage:{field}",
          ("context",), status="not_implemented",
          missing="output lineage (specification 9.3), including the mapper compatibility "
                  "of {Y/N} spellings"),
    _rule("P11", "the code is in its reference list (ISO 3166-1, ISO 4217, NUTS level 3 of "
          "the classification year)", "E", "Warn", "reference_list:{field}",
          ("violation", "limitation"), counts=True,
          source=f"{_ITS}; {_WORKBOOK}: valid ISO code, valid NUTS code; Q&A 1313",
          needs=("the bundled list for the classification year",),
          status="partly_implemented",
          missing="whether an LEI belongs to the named entity needs the GLEIF register, "
                  "which is outside the two files"),
    _rule("P12", "plausibility window", "P", "Warn", "plausibility:{field}", ("question",),
          applies_to="valid dates; percentage columns; RREC8"),
    _rule("P13", "the currency of the amount is stated", "E", "Warn",
          "currency_not_stated:{field}", ("violation", "context"),
          source=f"{_WORKBOOK}: the currency field must contain a valid ISO currency code; "
                 f"{_RTS}, Annex 2 ({{MONETARY}} definitions)",
          applies_to="present amounts of every {MONETARY} field",
          needs=("the amount parses as a number",), status="partly_implemented",
          missing="case 3 (the companion and the dedicated field differ) is C30's"),
)

# Rules of the later releases, by family (specification section 7). Each is
# recorded so the report lists it as not assessed.
_LATER: dict[str, tuple[str, Basis, Severity, tuple[str, ...]]] = {
    "A": ("the files (structure, duplicates, cut-off, identifiers)", "L", "Stop",
          tuple(f"A{n}" for n in range(1, 18))),
    "C": ("consistency within a loan record", "X", "Warn",
          tuple(f"C{n}" for n in range(1, 48) if n != 29)),
    "D": ("a borrower's records", "X", "Warn", ("D1", "D2", "D3")),
    "E": ("collateral records, and collateral against the loan", "X", "Warn",
          tuple(f"E{n}" for n in range(1, 20) if n != 14)),
    "F": ("the relationship between the files", "L", "Warn",
          tuple(f"F{n}" for n in range(1, 11))),
    "G": ("pool level", "P", "Warn", ("G1", "G2", "G3", "G4", "G6", "G7", "G8")),
    "L": ("consistency with the claimed lifecycle", "E", "Warn", ("L1",)),
}

RULES: dict[str, RuleEntry] = {r.rule_id: r for r in _P_RULES} | {
    rule_id: _rule(
        rule_id, f"family {family}: {title}", basis, severity, f"{rule_id}", _V,
        status="not_implemented", missing=f"family {family} arrives in a later release",
        applies_to="see specification section 7",
    )
    for family, (title, basis, severity, ids) in _LATER.items()
    for rule_id in ids
}

# What a finding group says, one full clause per rule and result kind
# (specification 3.2). The wording follows the result kind first and the
# basis second: only a violation of a reporting requirement (basis E)
# "does not comply with" something; a violation with basis X
# "contradicts"; a pipeline-constraint row (basis L) says what the pipeline
# does with the value; a question asks for confirmation and a plausibility
# result never asserts an error; a limitation says what was not assessed.
# The summary prints "<field>: <clause>."
SENTENCES: dict[tuple[str, str], str] = {
    ("P1", "violation"):
        "does not comply with the requirement to report every template field: the file has "
        "no column for this field",
    ("P3a", "violation"):
        "does not comply with Annex 2: the cell holds ND1, ND2, ND3 or ND4, which the field "
        "does not allow",
    ("P3b", "violation"):
        "does not comply with Annex 2: the cell holds ND5, which the field does not allow",
    ("P3c", "violation"):
        "does not comply with the requirement to report a value or a permitted ND option: "
        "the cell is empty",
    ("P3d", "violation"):
        "does not comply with the requirement to report a value or a permitted ND option: "
        "the cell holds a missing-value marker that is not one of ND1 to ND5",
    ("P3e", "violation"):
        "does not comply with Article 9(3): the ND4 is not written ND4-YYYY-MM-DD with a real "
        "date after the data cut-off date",
    ("P3f", "violation"):
        "does not comply with the field definition: the collateral is not property, so ND5 "
        "is the prescribed entry, and the cell holds a value",
    ("P3f", "question"):
        "may not comply; please confirm: the cell holds an ND option beside a date and the "
        "mixture is not explained",
    ("P3f", "limitation"):
        "could not be assessed: whether ND5 is the right entry depends on the collateral type "
        "or on the lifecycle of the loan, which this release does not establish for the record",
    ("P3g", "violation"):
        "the pipeline cannot read the file as it is: its loader does not recognise "
        "ND4-YYYY-MM-DD in a numeric field and stops at this value",
    ("P3g", "context"):
        "the pipeline's loader reads the cell differently from what the file states; the "
        "reason on each row says how",
    ("P4(a)", "accepted alternative"):
        "the value is valid with the whitespace around it, which the XSD type collapses; this "
        "application's CSV convention is the value without it",
    ("P4(c)", "violation"):
        "does not comply with the format prescribed for the field: the value holds a control "
        "character or a line break",
    ("P4(d)", "violation"):
        "the pipeline compares this code exactly and does not recognise the value as written; "
        "it is a code of the list only after trimming or upper-casing",
    ("P5", "violation"):
        "does not comply with the format prescribed for the field; the reason on each row says "
        "how",
    ("P5", "accepted alternative"):
        "the value is a valid XML boolean (true, false, 1 or 0); the template form is Y or N",
    ("P6(a)", "violation"):
        "does not comply with the published rule that the amount is not below 0",
    ("P6(b)", "violation"):
        "contradicts the published rule that the date is not later than the data cut-off date",
    ("P6(c)", "violation"):
        "does not comply with the published rule that the value is between 0 and 100",
    ("P6(d)", "violation"):
        "does not comply with the published rule that the year is greater than 1980 and less "
        "than 2100",
    ("P6(e)", "violation"):
        "does not comply with the published rule that the data cut-off date is in the past",
    ("P7(a)", "violation"):
        "does not comply with the maximum length of the field",
    ("P7(b)", "violation"):
        "does not comply with the text format of the technical standard: the value has a "
        "lower-case letter or a character outside ASCII (the XSD itself would accept it)",
    ("P7(c)", "violation"):
        "does not comply with Article 11(1): the value is not the reporting entity's LEI, the "
        "letter N, a four-digit year and a two-digit number",
    ("P8", "question"):
        "possible placeholder; confirm whether this is a measurement",
    ("P9", "context"):
        "the column holds one value on every record that reports a value",
    ("P11", "violation"):
        "does not comply with the requirement of a valid code: the value is not in its "
        "reference list (ISO 3166-1, ISO 4217, or NUTS level 3 of the classification year)",
    ("P11", "limitation"):
        "could not be assessed in full: the code has the right form, and its membership was "
        "not tested because the classification year is not established or not bundled, or "
        "the bundled year does not cover the country",
    ("P12", "question"):
        "unusual; please confirm: the value is outside the plausibility window given on each "
        "row",
    ("P13", "violation"):
        "does not comply with the requirement that the currency of every amount is stated: "
        "neither the companion column nor the dedicated field gives a valid ISO 4217 code",
    ("P13", "context"):
        "the file has no currency column for this amount, so the currency is taken from the "
        "other permitted evidence; this is a note about this application's CSV layout",
}


def sentence_clause(rule_id: str, kind: str) -> str:
    """The clause for a finding group. Every (rule, kind) an implemented
    rule can emit is in `SENTENCES` (a test requires it); the fallbacks
    serve the rules of later releases, which emit nothing yet."""
    clause = SENTENCES.get((rule_id, kind))
    if clause is not None:
        return clause
    rule = RULES[rule_id]
    if kind == "violation":
        if rule.basis == "E":
            return f"does not comply with a reporting requirement (rule {rule_id})"
        if rule.basis == "X":
            return f"contradicts another supplied value (rule {rule_id})"
        return f"the pipeline cannot use the value as written (rule {rule_id})"
    if kind == "question":
        if rule.basis == "P":
            return f"unusual; please confirm (rule {rule_id})"
        return f"may not comply; please confirm (rule {rule_id})"
    if kind == "limitation":
        return f"could not be assessed (rule {rule_id})"
    if kind == "accepted alternative":
        return f"the value is valid in a form other than the template's (rule {rule_id})"
    return f"an observation of rule {rule_id}"


# Statements about what two current-period files cannot show (specification
# 4.1). Printed once on every report; they never lower completeness.
SCOPE_STATEMENTS: tuple[str, ...] = (
    "Whether ND1 rather than ND2 or ND3 is the true reason for a missing value.",
    "Whether an ND5 is justified where no other field establishes applicability: "
    "allowed; justification not verifiable from the files.",
    "Identifier stability across reporting periods, and the history rules of the "
    "validation workbook.",
    "Whether an LEI belongs to the named entity, and whether a name matches the GLEIF "
    "register.",
    "Whether a NUTS code is the obligor's or the collateral's true region.",
    "Anything in the other Annex 2 sections or the investor report.",
    "Links between collateral records and loan records are not built in this release; "
    "they arrive with the identifier checks.",
)


def mapping_tokens(entry: FieldEntry) -> frozenset[str]:
    """The rule tokens the mapping file's `implementing_checks` cell holds
    for this field. `P4` stands for P4(a) to P4(d) and `P7` for P7(a) and
    P7(b), as the file writes them."""
    cells = DQ_POSSIBLE_PLACEHOLDERS["cells"]
    assert isinstance(cells, dict)
    placeholder_fields = set(cells) | {
        code
        for key in ("all_zero_columns", "single_token_columns")
        for code in DQ_POSSIBLE_PLACEHOLDERS[key]
    }
    tokens = {"P2", "P3a", "P3b", "P3c", "P3d", "P3e", "P3g", "P4", "P9"}
    tokens.add("O1" if entry.optional else "P1")
    if entry.kind == "ALPHANUM":
        tokens.add("P7")
        if entry.code in ("RREL1", "RREC1"):
            tokens.add("P7(c)")
    else:
        tokens.add("P5")
    if entry.code in CONDITIONAL_ND5_FIELDS or entry.code == "RREL65":
        tokens.add("P3f")
    if entry.kind == "MONETARY":
        tokens |= {"P6(a)", "P13"}
    if entry.not_after_cutoff:
        tokens.add("P6(b)")
    if entry.code == "RREL59":
        tokens.add("P6(c)")
    if entry.code == "RREL12":
        tokens.add("P6(d)")
    if entry.code == "RREL6":
        tokens |= {"P6(e)", "A10(a)"}
    if entry.code in placeholder_fields:
        tokens.add("P8")
    if entry.kind in ("NUTS", "COUNTRY", "CURRENCY") or entry.code == "RREL12":
        tokens.add("P11")
    if (
        entry.kind == "DATE"
        or entry.code == "RREC8"
        or (entry.kind == "PERCENTAGE" and entry.code not in DQ_PERCENT_SCALE_EXCLUDED_FIELDS)
    ):
        tokens.add("P12")
    return frozenset(tokens)
