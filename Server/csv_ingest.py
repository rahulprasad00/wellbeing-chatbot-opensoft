import csv
import io
import re
from sqlalchemy.orm import Session
from database.models import Master, HRUser, Conversation, Message, ActivityTracker, Leave, Onboarding, Performance, Rewards, Vibemeter
import json
from typing import Dict, Any

def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"

def parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def parse_int(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
    
def parse_shap_values(value: str):
    """
    Parse comma-separated feature names into a list of strings.
    e.g. "Performance_Rating,Promotion_Consideration,Leave_Type"
    -> ["Performance_Rating", "Promotion_Consideration", "Leave_Type"]
    """
    if not value:
        return []
    # Split by comma and strip whitespace
    return [item.strip() for item in value.split(",")]

def parse_shap_features(feature_string: str) -> Dict[str, Any]:
    """Parse SHAP feature string into dictionary"""
    features = {}
    if not feature_string:
        return features
    
    # This regex matches feature names and their values in parentheses
    pattern = r'([A-Za-z_]+)\(([^)]+)\)'
    matches = re.findall(pattern, feature_string)
    
    for feature_name, value in matches:
        # Optional: Trim any extra whitespace
        feature_name = feature_name.strip()
        value = value.strip()
        try:
            # Log the feature name and value (or remove after debugging)
            # Convert to float if it contains a dot, otherwise int
            if '.' in value:
                features[feature_name] = float(value)
            else:
                features[feature_name] = int(value)
        except ValueError:
            # If conversion fails, keep as string
            features[feature_name] = value
    
    return features

def parse_shap_nature(feature_string: str) -> Dict[str, Any]:
    """Parse SHAP feature string into dictionary"""
    features = {}
    if not feature_string:
        return features
    
    # This regex matches feature names and their values in parentheses
    pattern = r'([A-Za-z_]+)\(([^)]+)\)'
    matches = re.findall(pattern, feature_string)
    
    for feature_name, value in matches:
        # Optional: Trim any extra whitespace
        feature_name = feature_name.strip()
        value = value.strip()
        try:
            # Log the feature name and value (or remove after debugging)
            # Convert to float if it contains a dot, otherwise int
            if value==1:
                features[feature_name] = "POSITIVE"
            elif value==-1:
                features[feature_name] = "NEGATIVE"
        except ValueError:
            # If conversion fails, keep as string
            features[feature_name] = value
    
    return features


def ingest_csv_data(file_content: bytes, table: str, db: Session):
    """
    Reads CSV data from the given bytes and ingests records into the specified table.
    Supported tables: 'master', 'hr', 'conversation', 'message', 'activity_tracker', 'leave', 'onboarding', 'performance', 'rewards', 'vibemeter'
    """
    decoded = file_content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    records = []
    table = table.lower()

    if table == "master":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            employee_id = row.get("employee_id")
            if not employee_id:
                raise ValueError("Missing employee_id")
            # Map CSV columns to fixed fields in the Employee model

            shap_str = row.get("shap_values", "")
            shap_list = parse_shap_values(shap_str)
            # shap_nature=row.get("shap_nature", "")
            # shap_nature_list = parse_shap_nature(shap_nature)

            employee = Master(
                employee_id=row.get("employee_id"),
                shap_values=shap_list,
                shap_nature=shap_nature_list,
                employee_name=row.get("employee_name", ""),
                employee_email=row.get("employee_email"),
                password=hash_password(row.get("password")),
                # feature_vector= row.get("feature_vector", []),
                is_selected=row.get("is_selected")=="TRUE",
                # role=row.get("role", "employee"),
                # report=row.get("report", ""),
                # sentimental_score=parse_int(row.get("sentimental_score", "0")),
                # is_resolved=parse_bool(row.get("is_resolved", "false")),

                # work_hours=parse_float(row.get("work_hours", "0.0")),
                # leave_days=parse_int(row.get("leave_days", "0")),
                # leave_type=row.get("leave_type", ""),
                # performance_rating=parse_int(row.get("performance_rating", "0")),
                # manager_feedback=row.get("manager_feedback", ""),
                # promotion_consideration=parse_bool(row.get("promotion_consideration", "false")),
                # reward_points=parse_int(row.get("reward_points", "0")),
                # award_type=row.get("award_type", ""),
                # team_messages_sent=parse_int(row.get("team_messages_sent", "0")),
                # vibe_score=parse_int(row.get("vibe_score", "0"))
            )
            employee_id=row.get("employee_id")
            user=db.query(Master).filter(Master.employee_id == employee_id).first()
            if user:
                # Update existing record
                user.shap_values = shap_list
                user.shap_nature = shap_nature_list
                user.is_selected=row.get("should_reach_out")=="TRUE"
                # user.feature_vector=row.get("feature_vector", [])
                # user.employee_name=row.get("employee_name", "")
                # user.employee_email=row.get("employee_email")
            db.commit()
            db.refresh(user)
            records.append(user)

            
    elif table == "hr":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            hr_user = HRUser(
                email=row.get("email"),
                # password=hash_password(row.get("password")),
                role=row.get("role", "admin")
            )
            db.add(hr_user)
            records.append(hr_user)
    elif table == "conversation":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            try:
                messages_arr = json.loads(row.get("messages", "[]"))
            except Exception:
                messages_arr = []
            conversation = Conversation(
                id=int(row.get("conv_id")),  # assuming conv_id is provided as integer
                employee_id=row.get("employee_id"),
                messages=messages_arr
            )
            db.add(conversation)
            records.append(conversation)
    elif table == "message":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            message = Message(
                id=int(row.get("message_id")),  # assuming message_id is provided as integer
                conv_id=int(row.get("conv_id")),
                content=row.get("content")
            )
            db.add(message)
            records.append(message)
    elif table == "activity_tracker":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            user=db.query(ActivityTracker).filter(ActivityTracker.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.teams_messages_sent = row.get("teams_messages_sent")
                user.emails_sent = row.get("emails_sent")
                user.work_hours = row.get("work_hours")
                user.meetings_attended = row.get("meetings_attended")
                user.date = row.get("date")
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                activity = ActivityTracker(
                employee_id=row.get("employee_id"),
                date=row.get("date"),
                teams_messages_sent = row.get("teams_messages_sent"),
                emails_sent = row.get("emails_sent"),
                work_hours = row.get("work_hours"),
                meetings_attended = row.get("meetings_attended")
                )
                db.add(activity)
                records.append(activity)
    elif table == "leave":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            user=db.query(Leave).filter(Leave.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.leave_type = row.get("leave_type")
                user.leave_start_date = row.get("leave_start_date")
                user.leave_end_date = row.get("leave_end_date")
                user.leave_days = row.get("leave_days")
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                leave = Leave(
                    employee_id=row.get("employee_id"),
                    leave_type=row.get("leave_type"),
                    leave_start_date=row.get("leave_start_date"),
                    leave_end_date=row.get("leave_end_date"),
                    leave_days=row.get("leave_days")
                )
                db.add(leave)
                records.append(leave)
    elif table == "onboarding":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            # onboarding = Onboarding(
            #     employee_id=row.get("employee_id"),
            #     joining_date=row.get("joining_date"),
            #     onboarding_feedback=row.get("onboarding_feedback", ""),
            #     mentor_assigned = parse_bool(row.get("mentor_assigned")),
            #     initial_training_completed=parse_bool(row.get("initial_training_completed"))
            # )
            # db.add(onboarding)
            # records.append(onboarding)
            user=db.query(Onboarding).filter(Onboarding.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.joining_date = row.get("joining_date")
                user.onboarding_feedback = row.get("onboarding_feedback")
                user.mentor_assigned = parse_bool(row.get("mentor_assigned"))
                user.initial_training_completed = parse_bool(row.get("initial_training_completed"))
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                onboarding = Onboarding(
                    employee_id=row.get("employee_id"),
                    joining_date=row.get("joining_date"),
                    onboarding_feedback=row.get("onboarding_feedback", ""),
                    mentor_assigned = parse_bool(row.get("mentor_assigned")),
                    initial_training_completed=parse_bool(row.get("initial_training_completed"))
                )
                db.add(onboarding)
                records.append(onboarding)
    elif table == "performance":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            # performance = Performance(
            #     employee_id=row.get("employee_id"),
            #     review_period=row.get("review_period"),
            #     performance_rating=parse_int(row.get("performance_rating", "0")),
            #     manager_feedback=row.get("manager_feedback", ""),
            #     promotion_consideration = parse_bool(row.get("promotion_consideration"))
            # )
            # db.add(performance)
            # records.append(performance)
            user=db.query(Performance).filter(Performance.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.review_period = row.get("review_period")
                user.performance_rating = parse_int(row.get("performance_rating", "0"))
                user.manager_feedback = row.get("manager_feedback")
                user.promotion_consideration = parse_bool(row.get("promotion_consideration"))
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                performance = Performance(
                    employee_id=row.get("employee_id"),
                    review_period=row.get("review_period"),
                    performance_rating=parse_int(row.get("performance_rating", "0")),
                    manager_feedback=row.get("manager_feedback", ""),
                    promotion_consideration = parse_bool(row.get("promotion_consideration"))
                )
                db.add(performance)
                records.append(performance)
    elif table == "rewards":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            # rewards = Rewards(
            #     employee_id=row.get("employee_id"),
            #     award_date=row.get("award_date"),
            #     award_type=row.get("award_type", ""),
            #     reward_points=parse_int(row.get("reward_points", "0")),
            # )
            # db.add(rewards)
            # records.append(rewards)
            user=db.query(Rewards).filter(Rewards.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.award_date = row.get("award_date")
                user.award_type = row.get("award_type")
                user.reward_points = parse_int(row.get("reward_points", "0"))
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                rewards = Rewards(
                    employee_id=row.get("employee_id"),
                    award_date=row.get("award_date"),
                    award_type=row.get("award_type", ""),
                    reward_points=parse_int(row.get("reward_points", "0"))
                )
                db.add(rewards)
                records.append(rewards)
    elif table == "vibemeter":
        for row in reader:
            row = {key.lower(): value for key, value in row.items()}
            # vibemeter = Vibemeter(
            #     employee_id=row.get("employee_id"),
            #     response_date=row.get("response_date"),
            #     emotion_zone=row.get("emotion_zone", ""),
            #     vibe_score=row.get("vibe_score")
            # )
            # db.add(vibemeter)
            # records.append(vibemeter)
            user=db.query(Vibemeter).filter(Vibemeter.employee_id == row.get("employee_id")).first()
            if user:
                # Update existing record
                user.response_date = row.get("response_date")
                user.emotion_zone = row.get("emotion_zone")
                user.vibe_score = parse_int(row.get("vibe_score", "0"))
                db.commit()
                db.refresh(user)
                records.append(user)
            else:
                vibemeter = Vibemeter(
                    employee_id=row.get("employee_id"),
                    response_date=row.get("response_date"),
                    emotion_zone=row.get("emotion_zone", ""),
                    vibe_score=parse_int(row.get("vibe_score", "0"))
                )
                db.add(vibemeter)
                records.append(vibemeter)
    else:

        raise ValueError("Invalid table specified for ingestion.")
    
    db.commit()
    return len(records)


def update_master_feature_vector(db: Session):
    """
    For every unique employee_id found in the six datasets, update the Master table.
    The feature_vector field in Master is updated to include the table names (features)
    where that employee appears.
    """
    # List of tuples (table_feature_name, ModelClass)
    tables_to_check = [
        ("activity_tracker", ActivityTracker),
        ("leave", Leave),
        ("onboarding", Onboarding),
        ("performance", Performance),
        ("rewards", Rewards),
        ("vibemeter", Vibemeter)
    ]
    
    # Dictionary to collect employee_id and associated features (table names)
    employee_features = {}
    for feature_name, model in tables_to_check:
        results = db.query(model.employee_id).distinct().all()
        for (emp_id,) in results:
            if not emp_id:
                continue
            if emp_id not in employee_features:
                employee_features[emp_id] = set()
            employee_features[emp_id].add(feature_name)
    
    # Update or create records in Master table
    for emp_id, features in employee_features.items():
        master_record = db.query(Master).filter(Master.employee_id == emp_id).first()
        if master_record:
            # Update the feature_vector by unioning existing features with new ones
            existing_features = set(master_record.feature_vector) if master_record.feature_vector else set()
            updated_features = existing_features.union(features)
            master_record.feature_vector = list(updated_features)
        else:
            # Create new Master record with defaults and the feature_vector
            new_master = Master(
                employee_id=emp_id,
                feature_vector=list(features)
            )
            db.add(new_master)
    
    db.commit()


def ingest_shap_values(file_content: bytes, db: Session) -> int:
    """
    Ingests SHAP values from a CSV file and updates the Master table.
    Parses the feature strings into a proper dictionary format.
    """
    try:
        # Decode the CSV content
        decoded = file_content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        
        # Validate required columns
        required_columns = {"employee_id", "aggregated_shap_features"}
        if not required_columns.issubset(reader.fieldnames):
            missing_cols = required_columns - set(reader.fieldnames)
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        updated_count = 0

        for row in reader:
            employee_id = row["employee_id"].strip()
            features = parse_shap_features(row["aggregated_shap_features"])
            # print(f"Features for {employee_id}: {features}")
            
            # Convert is_selected to boolean
            # is_selected = row["is_selected"].strip().lower() == "true"
            # print(f"Is selected for {employee_id}: {is_selected}")

            # Query the existing record by employee_id
            master_record = db.query(Master).filter(Master.employee_id == employee_id).first()
            # print(f"Master record for {employee_id}: {master_record}")

            if master_record:
                # Update the existing record (assuming column name is shap_values)
                master_record.shap_values = features
                print(f"Updated shap_values for {employee_id}")
                # master_record.is_selected = is_selected
                updated_count += 1
                print(f"Updated record for {employee_id}")
            else:
                # Create a new record if it doesn't exist
                print(f"Creating new record for {employee_id}")
                new_record = Master(
                    employee_id=employee_id,
                    new_shap_values=features,
                    # is_selected=is_selected
                )
                db.add(new_record)
                updated_count += 1

        # Commit all changes
        print(f"Committing changes to database.")
        db.commit()
        return updated_count
    except Exception as e:
        db.rollback()
        raise Exception(f"Error ingesting SHAP values: {str(e)}")

# import psycopg2
# import csv

# # AWS RDS PostgreSQL connection details
# HOST = "opensoft.c9w2i6somhki.ap-south-1.rds.amazonaws.com"
# DATABASE = "postgres"   # Replace with your actual database name
# USER = "datamaster"
# PASSWORD = "databaseforever"
# CSV_FILE = "./cleaned_employee_data.csv"

# # Connect to PostgreSQL
# try:
#     conn = psycopg2.connect(
#         host=HOST,
#         database=DATABASE,
#         user=USER,
#         password=PASSWORD
#     )
#     print("✅ Connected to PostgreSQL")

#     cursor = conn.cursor()

#     # Ensure the table exists (create it if it doesn't)
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS employees (
#         Employee_ID TEXT PRIMARY KEY,
#         Work_Hours FLOAT,
#         Leave_Days INT,
#         Leave_Type TEXT,
#         Performance_Rating INT,
#         Manager_Feedback TEXT,
#         Promotion_Consideration BOOLEAN,
#         Reward_Points INT,
#         Award_Type TEXT,
#         Team_Messages_Sent INT,
#         Vibe_Score INT,
#         shap TEXT
#     );
#     """)
#     print("✅ Table checked/created")

#     # Open CSV file and import data
#     with open(CSV_FILE, 'r') as f:
#         reader = csv.reader(f)
#         header = next(reader)  # Skip header row
#         expected_cols = 12

#         for row in reader:
#             # Validate row length before inserting
#             if len(row) == expected_cols:
#                 cursor.execute("""
#                 INSERT INTO employees (
#                     Employee_ID, Work_Hours, Leave_Days, Leave_Type, Performance_Rating,
#                     Manager_Feedback, Promotion_Consideration, Reward_Points, Award_Type,
#                     Team_Messages_Sent, Vibe_Score, shap
#                 ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 """, row)
#             else:
#                 print(f"⚠️ Skipping invalid row: {len(row)} (length: {len(row)})")

#     conn.commit()
#     print("✅ CSV data imported successfully!")

# except Exception as e:
#     print(f"🔥 Error: {e}")
# finally:
#     cursor.close()
#     conn.close()
#     print("🔌 PostgreSQL connection closed")