import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = BASE_DIR

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
CORS(app)

app.config["JWT_SECRET_KEY"] = "my-super-secret-key-2026"
app.config["JWT_ALGORITHM"] = "HS256"
app.config["JWT_EXPIRE_HOURS"] = 168  # 7 天

# 临时写在代码中，正式环境请改为环境变量
ALIYUN_API_KEY = "sk-5ab598d9ec2541eaa01198f91c1dfa24"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"

DATABASE = os.path.join(BASE_DIR, "startup_evaluator.db")

SYSTEM_PROMPT = """你是一位专业的创业投资顾问。请根据用户提供的创业想法，撰写一份结构化评估报告。

报告须包含以下五个模块，分析时请在内心按模块组织，但最终只输出 JSON：
1. 市场痛点
2. 竞品分析
3. 用户画像
4. 风险提示
5. 下一步建议

输出要求：
- 必须严格只输出一个合法的 JSON 对象，禁止输出任何解释、前缀、后缀、markdown 代码块或多余文字
- JSON 键名固定且仅允许这五个：market、competition、user、risk、next
- 各字段值为对应模块的正文（约 150–300 字），内容具体、可执行，使用中文
- 正文中不要重复写「市场痛点」等模块标题（页面已有标题）

JSON 格式示例：
{
  "market": "...",
  "competition": "...",
  "user": "...",
  "risk": "...",
  "next": "..."
}"""

REQUIRED_KEYS = ("market", "competition", "user", "risk", "next")

USERS_TABLE_SQL = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);
"""

REPORTS_TABLE_SQL = """
CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    idea TEXT NOT NULL,
    market TEXT NOT NULL,
    competition TEXT NOT NULL,
    "user" TEXT NOT NULL,
    risk TEXT NOT NULL,
    next TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""


def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exception):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def table_exists(db, table_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def init_db():
    db = get_db()
    if not table_exists(db, "users"):
        db.executescript(USERS_TABLE_SQL)
    if not table_exists(db, "reports"):
        db.executescript(REPORTS_TABLE_SQL)
    db.commit()


def get_client() -> OpenAI:
    return OpenAI(api_key=ALIYUN_API_KEY, base_url=DASHSCOPE_BASE_URL)


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def call_bailian_evaluate(idea: str) -> dict:
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请评估以下创业想法：\n\n{idea}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("模型未返回有效内容")

    result = extract_json_object(content)
    missing = [key for key in REQUIRED_KEYS if key not in result]
    if missing:
        raise ValueError(f"模型返回缺少字段: {', '.join(missing)}")

    return {key: str(result[key]) for key in REQUIRED_KEYS}


def create_access_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=app.config["JWT_EXPIRE_HOURS"])
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(
        payload,
        app.config["JWT_SECRET_KEY"],
        algorithm=app.config["JWT_ALGORITHM"],
    )
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        app.config["JWT_SECRET_KEY"],
        algorithms=[app.config["JWT_ALGORITHM"]],
    )


def get_bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() or None
    return None


def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_bearer_token()
        if not token:
            return jsonify({"error": "未提供认证令牌，请先登录"}), 401
        try:
            payload = decode_access_token(token)
            g.current_user_id = int(payload["sub"])
            g.current_username = payload.get("username", "")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "登录已过期，请重新登录"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "无效的认证令牌"}), 401
        return f(*args, **kwargs)

    return decorated


def get_jwt_identity() -> int | None:
    """返回当前 JWT 中的用户 ID（须先通过 jwt_required 认证）。"""
    return getattr(g, "current_user_id", None)


def save_report(user_id: int, idea: str, result: dict) -> int:
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO reports (user_id, idea, market, competition, "user", risk, next)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            idea,
            result["market"],
            result["competition"],
            result["user"],
            result["risk"],
            result["next"],
        ),
    )
    db.commit()
    return cursor.lastrowid


def idea_summary(idea: str, max_len: int = 50) -> str:
    text = (idea or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


BEIJING_TZ = timezone(timedelta(hours=8))


def _parse_utc_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    if "+" in text[10:] or "-" in text[10:]:
        return datetime.fromisoformat(text)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {value}")


def format_created_at_beijing(value: str | None) -> str:
    """将 UTC 时间转为北京时间（UTC+8），格式 YYYY-MM-DD HH:MM:SS。"""
    if not value:
        return ""
    try:
        dt = _parse_utc_datetime(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(value)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400
        if len(username) < 3:
            return jsonify({"error": "用户名至少 3 个字符"}), 400
        if len(password) < 6:
            return jsonify({"error": "密码至少 6 个字符"}), 400

        password_hash = generate_password_hash(password)
        db = get_db()
        try:
            cursor = db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            db.commit()
        except sqlite3.IntegrityError:
            return jsonify({"error": "用户名已存在"}), 409

        user_id = cursor.lastrowid
        access_token = create_access_token(user_id, username)
        return jsonify(
            {
                "message": "注册成功",
                "user_id": user_id,
                "username": username,
                "token": access_token,
                "access_token": access_token,
            }
        ), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400

        db = get_db()
        row = db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None or not check_password_hash(row["password_hash"], password):
            return jsonify({"error": "用户名或密码错误"}), 401

        access_token = create_access_token(row["id"], row["username"])
        return jsonify(
            {
                "message": "登录成功",
                "user_id": row["id"],
                "username": row["username"],
                "token": access_token,
                "access_token": access_token,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate", methods=["POST"])
@jwt_required
def evaluate():
    try:
        user_id = get_jwt_identity()
        if user_id is None:
            return jsonify({"error": "未登录，请先登录"}), 401

        data = request.get_json(silent=True) or {}
        idea = (data.get("idea") or "").strip()
        if not idea:
            return jsonify({"error": "请提供创业想法（idea 字段不能为空）"}), 400

        result = call_bailian_evaluate(idea)
        save_report(user_id, idea, result)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports", methods=["GET"])
@jwt_required
def list_reports():
    try:
        user_id = get_jwt_identity()
        if user_id is None:
            return jsonify({"error": "未登录，请先登录"}), 401

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 5, type=int)
        search = request.args.get("search", "", type=str)
        
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 5

        offset = (page - 1) * per_page

        db = get_db()
        
        # 构建查询语句和参数
        if search:
            search_pattern = f"%{search}%"
            total = db.execute(
                "SELECT COUNT(*) AS cnt FROM reports WHERE user_id = ? AND idea LIKE ?",
                (user_id, search_pattern),
            ).fetchone()["cnt"]
            
            rows = db.execute(
                """
                SELECT id, idea, created_at
                FROM reports
                WHERE user_id = ? AND idea LIKE ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, search_pattern, per_page, offset),
            ).fetchall()
        else:
            total = db.execute(
                "SELECT COUNT(*) AS cnt FROM reports WHERE user_id = ?",
                (user_id,),
            ).fetchone()["cnt"]

            rows = db.execute(
                """
                SELECT id, idea, created_at
                FROM reports
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, per_page, offset),
            ).fetchall()

        reports = [
            {
                "id": row["id"],
                "idea": row["idea"],
                "created_at": format_created_at_beijing(row["created_at"]),
                "idea_summary": idea_summary(row["idea"]),
            }
            for row in rows
        ]
        return jsonify(
            {
                "total": total,
                "page": page,
                "per_page": per_page,
                "reports": reports,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report/<int:report_id>", methods=["GET"])
@jwt_required
def get_report(report_id: int):
    try:
        user_id = get_jwt_identity()
        if user_id is None:
            return jsonify({"error": "未登录，请先登录"}), 401

        db = get_db()
        row = db.execute(
            """
            SELECT id, idea, market, competition, "user", risk, next, created_at
            FROM reports
            WHERE id = ? AND user_id = ?
            """,
            (report_id, user_id),
        ).fetchone()

        if row is None:
            return jsonify({"error": "报告不存在或无权访问"}), 404

        return jsonify(
            {
                "id": row["id"],
                "idea": row["idea"],
                "created_at": row["created_at"],
                "market": row["market"],
                "competition": row["competition"],
                "user": row["user"],
                "risk": row["risk"],
                "next": row["next"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report/<int:report_id>", methods=["DELETE"])
@jwt_required
def delete_report(report_id: int):
    try:
        user_id = get_jwt_identity()
        if user_id is None:
            return jsonify({"success": False, "error": "未登录，请先登录"}), 401

        db = get_db()
        row = db.execute(
            "SELECT id, user_id FROM reports WHERE id = ?",
            (report_id,),
        ).fetchone()

        if row is None:
            return jsonify({"success": False, "error": "报告不存在"}), 404

        if row["user_id"] != user_id:
            return jsonify({"success": False, "error": "无权删除该报告"}), 403

        db.execute(
            "DELETE FROM reports WHERE id = ? AND user_id = ?",
            (report_id, user_id),
        )
        db.commit()

        return jsonify(
            {
                "success": True,
                "message": "报告已删除",
                "id": report_id,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


with app.app_context():
    init_db()


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
