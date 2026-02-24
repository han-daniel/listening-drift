from sqlalchemy import create_engine, text

engine = create_engine("postgresql://danielhan@localhost/music_behavior")

with engine.begin() as conn:
    conn.execute(
        text("INSERT INTO users (username, country, total_scrobbles) VALUES (:u, :c, :s) ON CONFLICT (username) DO NOTHING"),
        {"u": "test_user", "c": "US", "s": 1000},
    )

    result = conn.execute(text("SELECT * FROM users"))
    for row in result:
        print(row)
