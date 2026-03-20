"""
generate_mock_mimic.py - Generate MIMIC-IV mock data for DuckDB import testing

Purpose:
  When real MIMIC-IV data is unavailable, generate structurally identical
  small-scale CSV files to end-to-end verify the full import + concepts pipeline.

  Output goes to mimic-iv-mock/hosp/ and mimic-iv-mock/icu/,
  with directory structure matching the real MIMIC-IV download.

Data scale:
  - 20 patients (subject_id: 10001-10020)
  - 1 admission each (hadm_id: 20001-20020)
  - 15 with ICU stays (stay_id: 30001-30015)
  - 48 hours of vitals + labs per ICU patient
  - Includes antibiotics, vasopressors, mechanical ventilation
  - Includes microbiology cultures (for Sepsis-3 suspicion of infection)
"""
from __future__ import annotations

import gzip
import csv
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

MOCK_DIR = Path(__file__).resolve().parent.parent / "mimic-iv-mock"
N_PATIENTS = 20
N_ICU_PATIENTS = 15
ICU_HOURS = 48
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

BASE_TIME = datetime(2150, 1, 1, 8, 0, 0)


def ts(dt):
    """Format timestamp to MIMIC-IV format."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def write_csv_gz(filepath, header, rows):
    """Write gzip-compressed CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {filepath.name}: {len(rows)} rows")


# ============================================================
# Table Generators
# ============================================================

def gen_patients():
    """mimiciv_hosp.patients"""
    header = ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]
    rows = []
    for i in range(N_PATIENTS):
        sid = 10001 + i
        gender = random.choice(["M", "F"])
        age = random.randint(40, 85)
        dod = ""
        if i < N_ICU_PATIENTS and random.random() < 0.2:
            dod = (BASE_TIME + timedelta(hours=i * 72, days=random.randint(1, 28))).strftime("%Y-%m-%d")
        rows.append([sid, gender, age, 2150, "2147 - 2152", dod])
    write_csv_gz(MOCK_DIR / "hosp" / "patients.csv.gz", header, rows)


def gen_admissions():
    """mimiciv_hosp.admissions"""
    header = [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "admission_type", "admit_provider_id", "admission_location",
        "discharge_location", "insurance", "language", "marital_status",
        "race", "edregtime", "edouttime", "hospital_expire_flag"
    ]
    rows = []
    for i in range(N_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        admit = BASE_TIME + timedelta(hours=i * 72)
        disch = admit + timedelta(days=random.randint(3, 14))
        deathtime = ""
        expire_flag = 0
        if i < N_ICU_PATIENTS and random.random() < 0.2:
            deathtime = ts(disch - timedelta(hours=random.randint(1, 12)))
            expire_flag = 1
        rows.append([
            sid, hadm, ts(admit), ts(disch), deathtime,
            "EMERGENCY", f"P{random.randint(1000,9999)}", "EMERGENCY ROOM",
            "HOME" if not expire_flag else "DIED",
            "Medicare", "ENGLISH", random.choice(["MARRIED", "SINGLE", ""]),
            "WHITE", ts(admit - timedelta(hours=1)), ts(admit), expire_flag
        ])
    write_csv_gz(MOCK_DIR / "hosp" / "admissions.csv.gz", header, rows)


def gen_d_labitems():
    """mimiciv_hosp.d_labitems"""
    header = ["itemid", "label", "fluid", "category"]
    items = [
        (50912, "Creatinine", "Blood", "Chemistry"),
        (50813, "Lactate", "Blood", "Blood Gas"),
        (51265, "Platelet Count", "Blood", "Hematology"),
        (51301, "White Blood Cells", "Blood", "Hematology"),
        (50885, "Bilirubin, Total", "Blood", "Chemistry"),
        (50821, "pO2", "Blood", "Blood Gas"),
        (50816, "FiO2", "Blood", "Blood Gas"),
        (51237, "INR(PT)", "Blood", "Hematology"),
        (51222, "Hemoglobin", "Blood", "Hematology"),
        (50862, "Albumin", "Blood", "Chemistry"),
        (50882, "Bicarbonate", "Blood", "Chemistry"),
        (50902, "Chloride", "Blood", "Chemistry"),
        (50931, "Glucose", "Blood", "Chemistry"),
        (50971, "Potassium", "Blood", "Chemistry"),
        (50983, "Sodium", "Blood", "Chemistry"),
        (51006, "Urea Nitrogen", "Blood", "Chemistry"),
    ]
    rows = [[iid, label, fluid, cat] for iid, label, fluid, cat in items]
    write_csv_gz(MOCK_DIR / "hosp" / "d_labitems.csv.gz", header, rows)


def gen_labevents():
    """mimiciv_hosp.labevents"""
    header = [
        "labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
        "order_provider_id", "charttime", "storetime", "value", "valuenum",
        "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"
    ]
    rows = []
    lab_id = 100000
    lab_specs = [
        (50912, 0.6, 1.8, "mg/dL"),
        (50813, 0.5, 5.0, "mmol/L"),
        (51265, 100, 400, "K/uL"),
        (51301, 4.0, 20.0, "K/uL"),
        (50885, 0.3, 4.0, "mg/dL"),
        (51237, 0.8, 2.5, ""),
    ]
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        admit = BASE_TIME + timedelta(hours=i * 72)
        for hour in range(0, ICU_HOURS, 6):
            ct = admit + timedelta(hours=hour)
            spec_id = lab_id
            for itemid, low, high, uom in lab_specs:
                val = round(random.uniform(low, high), 2)
                lab_id += 1
                rows.append([
                    lab_id, sid, hadm, spec_id, itemid,
                    "", ts(ct), ts(ct + timedelta(minutes=30)),
                    str(val), val, uom, "", "", "", "ROUTINE", ""
                ])
    write_csv_gz(MOCK_DIR / "hosp" / "labevents.csv.gz", header, rows)


def gen_microbiologyevents():
    """mimiciv_hosp.microbiologyevents"""
    header = [
        "microevent_id", "subject_id", "hadm_id", "micro_specimen_id",
        "order_provider_id", "chartdate", "charttime", "spec_itemid",
        "spec_type_desc", "test_seq", "storedate", "storetime",
        "test_itemid", "test_name", "org_itemid", "org_name",
        "isolate_num", "quantity", "ab_itemid", "ab_name",
        "dilution_text", "dilution_comparison", "dilution_value",
        "interpretation", "comments"
    ]
    rows = []
    micro_id = 500000
    for i in range(N_ICU_PATIENTS):
        if random.random() < 0.7:
            sid = 10001 + i
            hadm = 20001 + i
            ct = BASE_TIME + timedelta(hours=i * 72 + random.randint(0, 12))
            micro_id += 1
            positive = random.random() < 0.4
            rows.append([
                micro_id, sid, hadm, micro_id,
                "", ts(ct), ts(ct), 70012,
                "BLOOD CULTURE", 1, ts(ct), ts(ct),
                90003, "AEROBIC BOTTLE",
                80004 if positive else "", "STAPH AUREUS COAG +" if positive else "",
                1 if positive else "", "", "", "",
                "", "", "", "", ""
            ])
    write_csv_gz(MOCK_DIR / "hosp" / "microbiologyevents.csv.gz", header, rows)


def gen_prescriptions():
    """mimiciv_hosp.prescriptions"""
    header = [
        "subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq",
        "order_provider_id", "starttime", "stoptime", "drug_type", "drug",
        "formulary_drug_cd", "gsn", "ndc", "prod_strength", "form_rx",
        "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp",
        "doses_per_24_hrs", "route"
    ]
    rows = []
    antibiotics = ["Vancomycin", "Piperacillin-Tazobactam", "Cefepime", "Meropenem"]
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        admit = BASE_TIME + timedelta(hours=i * 72)
        if random.random() < 0.8:
            ab_start = admit + timedelta(hours=random.randint(0, 6))
            ab = random.choice(antibiotics)
            rows.append([
                sid, hadm, 60000 + i, "", "",
                "", ts(ab_start), ts(ab_start + timedelta(days=5)),
                "MAIN", ab,
                "", "", "", "", "",
                "1", "g", "", "",
                3, "IV"
            ])
    write_csv_gz(MOCK_DIR / "hosp" / "prescriptions.csv.gz", header, rows)


def gen_icustays():
    """mimiciv_icu.icustays"""
    header = [
        "subject_id", "hadm_id", "stay_id", "first_careunit",
        "last_careunit", "intime", "outtime", "los"
    ]
    rows = []
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        stay = 30001 + i
        intime = BASE_TIME + timedelta(hours=i * 72 + 1)
        los_days = random.uniform(1, 8)
        outtime = intime + timedelta(days=los_days)
        rows.append([
            sid, hadm, stay,
            random.choice(["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)"]),
            random.choice(["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)"]),
            ts(intime), ts(outtime), round(los_days, 4)
        ])
    write_csv_gz(MOCK_DIR / "icu" / "icustays.csv.gz", header, rows)


def gen_d_items():
    """mimiciv_icu.d_items"""
    header = [
        "itemid", "label", "abbreviation", "linksto",
        "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"
    ]
    items = [
        (220045, "Heart Rate", "HR", "chartevents", "Routine Vital Signs", "bpm", "Numeric", 60, 100),
        (220179, "Non Invasive Blood Pressure systolic", "NBPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 90, 140),
        (220180, "Non Invasive Blood Pressure diastolic", "NBPd", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 60, 90),
        (220181, "Non Invasive Blood Pressure mean", "NBPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 65, 110),
        (220050, "Arterial Blood Pressure systolic", "ABPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 90, 140),
        (220051, "Arterial Blood Pressure diastolic", "ABPd", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 60, 90),
        (220052, "Arterial Blood Pressure mean", "ABPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 65, 110),
        (220210, "Respiratory Rate", "RR", "chartevents", "Routine Vital Signs", "/min", "Numeric", 12, 20),
        (220277, "O2 saturation pulseoxymetry", "SpO2", "chartevents", "Routine Vital Signs", "%", "Numeric", 95, 100),
        (223761, "Temperature Fahrenheit", "Temp F", "chartevents", "Routine Vital Signs", "deg F", "Numeric", 96.8, 100.4),
        (223762, "Temperature Celsius", "Temp C", "chartevents", "Routine Vital Signs", "deg C", "Numeric", 36, 38),
        (220739, "GCS - Eye Opening", "GCSEye", "chartevents", "Neurological", "", "Numeric", 1, 4),
        (223900, "GCS - Verbal Response", "GCSVerb", "chartevents", "Neurological", "", "Numeric", 1, 5),
        (223901, "GCS - Motor Response", "GCSMotor", "chartevents", "Neurological", "", "Numeric", 1, 6),
        (220235, "Inspired O2 Fraction", "FiO2", "chartevents", "Respiratory", "%", "Numeric", 21, 100),
        (225792, "Invasive Ventilation", "InvVent", "chartevents", "Respiratory", "", "Text", "", ""),
        (226732, "O2 Delivery Device(s)", "O2Dev", "chartevents", "Respiratory", "", "Text", "", ""),
        (221906, "Norepinephrine", "Norepi", "inputevents", "Medications", "mcg/kg/min", "Numeric", "", ""),
        (221662, "Dopamine", "Dopa", "inputevents", "Medications", "mcg/kg/min", "Numeric", "", ""),
        (221289, "Epinephrine", "Epi", "inputevents", "Medications", "mcg/kg/min", "Numeric", "", ""),
        (222315, "Vasopressin", "Vaso", "inputevents", "Medications", "units/min", "Numeric", "", ""),
        (221749, "Phenylephrine", "Phenyl", "inputevents", "Medications", "mcg/min", "Numeric", "", ""),
        (221986, "Milrinone", "Milri", "inputevents", "Medications", "mcg/kg/min", "Numeric", "", ""),
        (221712, "Dobutamine", "Dobut", "inputevents", "Medications", "mcg/kg/min", "Numeric", "", ""),
        (226559, "Foley", "Foley", "outputevents", "Output", "mL", "Numeric", "", ""),
    ]
    rows = [[iid, label, abbr, link, cat, unit, pt, lo, hi]
            for iid, label, abbr, link, cat, unit, pt, lo, hi in items]
    write_csv_gz(MOCK_DIR / "icu" / "d_items.csv.gz", header, rows)


def gen_chartevents():
    """mimiciv_icu.chartevents - Core: hourly vital signs"""
    header = [
        "subject_id", "hadm_id", "stay_id", "caregiver_id",
        "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"
    ]
    rows = []
    vitals = [
        (220045, 60, 130, "bpm"),
        (220179, 80, 160, "mmHg"),
        (220180, 50, 100, "mmHg"),
        (220052, 55, 110, "mmHg"),
        (220210, 10, 30, "/min"),
        (220277, 88, 100, "%"),
        (223762, 36.0, 39.5, "deg C"),
    ]
    gcs_items = [
        (220739, 1, 4, ""),
        (223900, 1, 5, ""),
        (223901, 1, 6, ""),
    ]
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        stay = 30001 + i
        icu_in = BASE_TIME + timedelta(hours=i * 72 + 1)

        for hour in range(ICU_HOURS):
            ct = icu_in + timedelta(hours=hour)
            for itemid, lo, hi, uom in vitals:
                val = round(random.uniform(lo, hi), 1)
                rows.append([sid, hadm, stay, "", ts(ct), ts(ct), itemid, str(val), val, uom, 0])

            if hour % 4 == 0:
                for itemid, lo, hi, uom in gcs_items:
                    val = random.randint(lo, hi)
                    rows.append([sid, hadm, stay, "", ts(ct), ts(ct), itemid, str(val), val, uom, 0])

    write_csv_gz(MOCK_DIR / "icu" / "chartevents.csv.gz", header, rows)


def gen_inputevents():
    """mimiciv_icu.inputevents - Vasopressor infusions"""
    header = [
        "subject_id", "hadm_id", "stay_id", "caregiver_id",
        "starttime", "endtime", "storetime", "itemid", "amount", "amountuom",
        "rate", "rateuom", "orderid", "linkorderid",
        "ordercategoryname", "secondaryordercategoryname",
        "ordercomponenttypedescription", "ordercategorydescription",
        "patientweight", "totalamount", "totalamountuom",
        "isopenbag", "continueinnextdept", "statusdescription",
        "originalamount", "originalrate"
    ]
    rows = []
    order_id = 700000
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        stay = 30001 + i
        icu_in = BASE_TIME + timedelta(hours=i * 72 + 1)
        if random.random() < 0.4:
            start_h = random.randint(2, 12)
            dur_h = random.randint(6, 36)
            start = icu_in + timedelta(hours=start_h)
            end = start + timedelta(hours=dur_h)
            rate = round(random.uniform(0.02, 0.3), 3)
            order_id += 1
            rows.append([
                sid, hadm, stay, "",
                ts(start), ts(end), ts(start), 221906,
                round(rate * dur_h * 60, 2), "mg",
                rate, "mcg/kg/min", order_id, order_id,
                "01-Drips", "", "Main order parameter", "Continuous Med",
                round(random.uniform(55, 100), 1), "", "",
                0, 0, "FinishedRunning",
                round(rate * dur_h * 60, 2), rate
            ])
    write_csv_gz(MOCK_DIR / "icu" / "inputevents.csv.gz", header, rows)


def gen_outputevents():
    """mimiciv_icu.outputevents - Urine output"""
    header = [
        "subject_id", "hadm_id", "stay_id", "caregiver_id",
        "charttime", "storetime", "itemid", "value", "valueuom"
    ]
    rows = []
    for i in range(N_ICU_PATIENTS):
        sid = 10001 + i
        hadm = 20001 + i
        stay = 30001 + i
        icu_in = BASE_TIME + timedelta(hours=i * 72 + 1)
        for hour in range(0, ICU_HOURS, 4):
            ct = icu_in + timedelta(hours=hour)
            vol = round(random.uniform(50, 400), 0)
            rows.append([sid, hadm, stay, "", ts(ct), ts(ct), 226559, vol, "mL"])
    write_csv_gz(MOCK_DIR / "icu" / "outputevents.csv.gz", header, rows)


def gen_procedureevents():
    """mimiciv_icu.procedureevents - Mechanical ventilation etc."""
    header = [
        "subject_id", "hadm_id", "stay_id", "caregiver_id",
        "starttime", "endtime", "storetime", "itemid", "value", "valueuom",
        "location", "locationcategory", "orderid", "linkorderid",
        "ordercategoryname", "ordercategorydescription",
        "patientweight", "isopenbag", "continueinnextdept",
        "statusdescription", "originalamount", "originalrate"
    ]
    rows = []
    order_id = 800000
    for i in range(N_ICU_PATIENTS):
        if random.random() < 0.3:
            sid = 10001 + i
            hadm = 20001 + i
            stay = 30001 + i
            icu_in = BASE_TIME + timedelta(hours=i * 72 + 1)
            start = icu_in + timedelta(hours=random.randint(0, 6))
            end = start + timedelta(hours=random.randint(12, 48))
            order_id += 1
            rows.append([
                sid, hadm, stay, "",
                ts(start), ts(end), ts(start), 225792, "", "",
                "", "", order_id, order_id,
                "Communication", "Communication",
                "", 0, 0, "FinishedRunning", "", ""
            ])
    write_csv_gz(MOCK_DIR / "icu" / "procedureevents.csv.gz", header, rows)


def gen_stub_tables():
    """Generate empty or minimal auxiliary tables required by DuckDB schema."""
    stubs = {
        "hosp": {
            "d_hcpcs.csv.gz": (["code", "category", "long_description", "short_description"], []),
            "diagnoses_icd.csv.gz": (["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"], []),
            "d_icd_diagnoses.csv.gz": (["icd_code", "icd_version", "long_title"], []),
            "d_icd_procedures.csv.gz": (["icd_code", "icd_version", "long_title"], []),
            "drgcodes.csv.gz": (["subject_id", "hadm_id", "drg_type", "drg_code", "description", "drg_severity", "drg_mortality"], []),
            "emar_detail.csv.gz": (["subject_id", "emar_id", "emar_seq", "parent_field_ordinal", "administration_type", "pharmacy_id", "barcode_type", "reason_for_no_barcode", "complete_dose_not_given", "dose_due", "dose_due_unit", "dose_given", "dose_given_unit", "will_remainder_of_dose_be_given", "product_amount_given", "product_unit", "product_code", "product_description", "product_description_other", "prior_infusion_rate", "infusion_rate", "infusion_rate_adjustment", "infusion_rate_adjustment_amount", "infusion_rate_unit", "route", "infusion_complete", "completion_interval", "new_iv_bag_hung", "continued_infusion_in_other_location", "restart_interval", "side", "site", "non_formulary_visual_verification"], []),
            "emar.csv.gz": (["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id", "enter_provider_id", "charttime", "medication", "event_txt", "scheduletime", "storetime"], []),
            "hcpcsevents.csv.gz": (["subject_id", "hadm_id", "chartdate", "hcpcs_cd", "seq_num", "short_description"], []),
            "omr.csv.gz": (["subject_id", "chartdate", "seq_num", "result_name", "result_value"], []),
            "pharmacy.csv.gz": (["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "medication", "proc_type", "status", "entertime", "verifiedtime", "route", "frequency", "disp_sched", "infusion_type", "sliding_scale", "lockout_interval", "basal_rate", "one_hr_max", "doses_per_24_hrs", "duration", "duration_interval", "expiration_value", "expiration_unit", "expirationdate", "dispensation", "fill_quantity"], []),
            "poe_detail.csv.gz": (["poe_id", "poe_seq", "subject_id", "field_name", "field_value"], []),
            "poe.csv.gz": (["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "transaction_type", "discontinue_of_poe_id", "discontinued_by_poe_id", "order_provider_id", "order_status"], []),
            "procedures_icd.csv.gz": (["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"], []),
            "provider.csv.gz": (["provider_id"], [["P1001"]]),
            "services.csv.gz": (["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"], []),
            "transfers.csv.gz": (["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"], []),
        },
        "icu": {
            "caregiver.csv.gz": (["caregiver_id"], [[1], [2], [3]]),
            "datetimeevents.csv.gz": (["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valueuom", "warning"], []),
            "ingredientevents.csv.gz": (["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime", "storetime", "itemid", "amount", "amountuom", "rate", "rateuom", "orderid", "linkorderid", "statusdescription", "originalamount", "originalrate"], []),
        },
    }
    for schema, tables in stubs.items():
        for filename, (header, rows) in tables.items():
            write_csv_gz(MOCK_DIR / schema / filename, header, rows)


# ============================================================
# Main
# ============================================================

def main():
    print(f"Generating MIMIC-IV mock data to {MOCK_DIR}")
    print(f"Config: {N_PATIENTS} patients, {N_ICU_PATIENTS} ICU, {ICU_HOURS}h window")
    print()

    print("[1/11] patients")
    gen_patients()
    print("[2/11] admissions")
    gen_admissions()
    print("[3/11] d_labitems")
    gen_d_labitems()
    print("[4/11] labevents")
    gen_labevents()
    print("[5/11] microbiologyevents")
    gen_microbiologyevents()
    print("[6/11] prescriptions")
    gen_prescriptions()
    print("[7/11] icustays")
    gen_icustays()
    print("[8/11] d_items")
    gen_d_items()
    print("[9/11] chartevents")
    gen_chartevents()
    print("[10/11] inputevents + outputevents + procedureevents")
    gen_inputevents()
    gen_outputevents()
    gen_procedureevents()
    print("[11/11] stub tables")
    gen_stub_tables()

    print()
    print("Done! Directory structure:")
    for schema in ["hosp", "icu"]:
        d = MOCK_DIR / schema
        files = sorted(d.glob("*.csv.gz"))
        print(f"  {schema}/: {len(files)} files")
        for f in files:
            size = f.stat().st_size
            print(f"    {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
