from . import db
from geoalchemy2 import Geometry

def create_Idataset_model(name):
    class Idataset(db.Model):
        __tablename__ = name
        __table_args__ = {
            'schema': 'idataset'
        }
        
        id = db.Column(db.String(64), primary_key=True)
        title = db.Column(db.String(64))
        file = db.Column(db.String(64))
        geom = db.Column(Geometry('POLYGON'))
        width = db.Column(db.String(64))
        height = db.Column(db.String(64))

        def __repr__(self):
            return '<Geom %r>' % self.geom
    return Idataset