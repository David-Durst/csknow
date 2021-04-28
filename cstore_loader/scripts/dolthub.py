from doltpy.cli import Dolt
from doltpy.sql import DoltSQLServerContext, ServerConfig
import sqlalchemy as sa
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
f = open(cur_dir + "../../private/.mysql_password", "r")
mysql_password = f.read().strip("\n")

# Setup objects to represents source and target databases, start Dolt SQL Server
dolt = Dolt.clone('durst/csknow')
dssc = DoltSQLServerContext(dolt, ServerConfig())
dssc.start_server()
mysql_engine = sa.create_engine(
    '{dialect}://{user}:{password}@{host}:{port}/{database}'.format(
        dialect='mysql+mysqlconnector',
        user="root",
        password=mysql_password,
        host="localhost",
        port="3124",
        database="csknow"
    )
)

from doltpy.sql.sync import sync_schema_to_dolt, MYSQL_TO_DOLT_TYPE_MAPPING

sync_schema_to_dolt(mysql_engine,
                    dssc.engine,
                    {"players":"players", "rounds":"rounds", "ticks":"ticks", "player_at_tick":"player_at_tick", "spotted":"spotted", "weapon_fire":"weapon_fire", "kills":"kills", "hurt":"hurt", "grenades":"grenades" "flashed":"flashed", "grenade_trajectories":"grenade_trajectories", "plants":"plants", "defusals":"defusals", "explosions":"explosions"},
                    MYSQL_TO_DOLT_TYPE_MAPPINGS)

dssc.stop_server()
