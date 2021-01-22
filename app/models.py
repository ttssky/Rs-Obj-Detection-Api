from . import db

class Idataset(db.Model):
    __tablename__ = '''t_37605aef-913e-40b6-8859-822e72a51a19'''
    __table_args__ = {
        'schema': 'idataset'
    }
    
    id = db.Column(db.String(64), primary_key=True)
    title = db.Column(db.String(64))
    file = db.Column(db.String(64))

    def __repr__(self):
        return '<File %r>' % self.file