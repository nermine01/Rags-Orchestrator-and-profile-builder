from alembic import op
import sqlalchemy as sa

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "agents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("agent_id", sa.Integer, sa.ForeignKey("agents.id"), nullable=False, index=True),
        sa.Column("mime", sa.String(100), server_default="application/pdf"),
        sa.Column("status", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("file_path", sa.String(1024)),
        sa.Column("file_data", sa.LargeBinary),
        sa.Column("index_path", sa.String(1024)),
        sa.Column("indexed_at", sa.DateTime),
        sa.Column("last_error", sa.Text),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("agent_id", sa.Integer, sa.ForeignKey("agents.id"), nullable=False, index=True),
        sa.Column("user_id", sa.String(255)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer, sa.ForeignKey("sessions.id"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),   # user|assistant|tool
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tool_name", sa.String(100)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )

def downgrade():
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("documents")
    op.drop_table("agents")
