from alembic import op
import sqlalchemy as sa

revision = "0002_add_agent_centroid"
down_revision = "0001_init"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("agents", sa.Column("centroid", sa.LargeBinary, nullable=True))

def downgrade():
    op.drop_column("agents", "centroid")
