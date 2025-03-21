from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_admin_uuid():
    """Fetch the UUID of the admin user from PostgreSQL."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id FROM users WHERE username = :username"), {"username": "admin"})
            admin_data = result.fetchone()
            if admin_data:
                return str(admin_data[0])  # Convert UUID to string
            else:
                raise ValueError("Admin user not found in database.")
    except Exception as e:
        raise ValueError(f"Error fetching admin UUID: {str(e)}")

@app.route('/api/tools', methods=['GET'])
def get_tools():
    """Get all tools."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM tools"))
            tools = [dict(row) for row in result.mappings()]
        return jsonify(tools)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/admin', methods=['GET'])
def get_admin_checked_out_tools():
    """Get tools checked out by the admin user."""
    try:
        admin_uuid = get_admin_uuid()

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM tools 
                WHERE status = 'checked_out' 
                AND checked_out_by = :user_id
            """), {"user_id": uuid.UUID(admin_uuid)})  # Ensure UUID format
            
            tools = [dict(row) for row in result.mappings()]
        return jsonify(tools)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/checkout', methods=['POST'])
def checkout_tool():
    """Check-out a tool."""
    try:
        data = request.json
        tool_id = data.get('tool_id')
        user_id = data.get('user_id')
        current_time = datetime.now().isoformat()

        if not tool_id or not user_id:
            return jsonify({'error': 'Tool ID and User ID are required'}), 400

        try:
            user_uuid = uuid.UUID(user_id)  # Validate UUID format
        except ValueError:
            if user_id == "admin":
                user_uuid = get_admin_uuid()  # Fetch the actual UUID of "admin"
                if not user_uuid:
                    return jsonify({'error': 'Admin user UUID not found'}), 400
            else:
                return jsonify({'error': 'Invalid UUID format for user_id'}), 400

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE tools 
                SET status = 'checked_out', 
                    checked_out_by = :user_id, 
                    checkout_time = :timestamp
                WHERE id = :tool_id AND status = 'available'
            """), {"tool_id": tool_id, "user_id": str(user_uuid), "timestamp": current_time})

            conn.execute(text("""
                INSERT INTO tool_logs (tool_id, user_id, action, timestamp)
                VALUES (:tool_id, :user_id, 'checkout', :timestamp)
            """), {"tool_id": tool_id, "user_id": str(user_uuid), "timestamp": current_time})

        return jsonify({'message': 'Tool checked out successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/tools/checkin', methods=['POST'])
def checkin_tool():
    """Check-in a tool."""
    try:
        data = request.json
        tool_id = data.get('tool_id')
        user_id = data.get('user_id')
        current_time = datetime.now().isoformat()

        if not tool_id or not user_id:
            return jsonify({'error': 'Tool ID and User ID are required'}), 400

        try:
            user_uuid = uuid.UUID(user_id)  # Ensure user_id is a valid UUID
        except ValueError:
            return jsonify({'error': 'Invalid UUID format for user_id'}), 400

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE tools 
                SET status = 'available', 
                    checked_out_by = NULL, 
                    checkout_time = NULL
                WHERE id = :tool_id
            """), {"tool_id": tool_id})

            conn.execute(text("""
                INSERT INTO tool_logs (tool_id, user_id, action, timestamp)
                VALUES (:tool_id, :user_id, 'checkin', :timestamp)
            """), {"tool_id": tool_id, "user_id": str(user_uuid), "timestamp": current_time})

        return jsonify({'message': 'Tool checked in successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

