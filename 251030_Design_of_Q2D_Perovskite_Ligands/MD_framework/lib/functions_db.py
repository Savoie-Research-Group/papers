#!/bin/env python
"""
Created on Wed Jun 10 12:49:35 2020

@author: Stephen Shiring

DB functions for SQLite
"""

import sqlite3

def create_connection(db_filename):
    connection = None
    try:
        connection = sqlite3.connect(db_filename)
    except sqlite3.Error as error:
        print(('\nERROR: Failed to connection to database "{}".\n       Error: {}\n'.format(error)))

    return connection

def create_table(connection, sql):
    try:
        c = connection.cursor()
        c.execute(sql)
    except sqlite3.Error as error:
        print(("\nERROR: Failed to create table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return False
    return True

def add_body(connection, Body, commit=False):
    sql = ''' INSERT INTO bodies(name, chem_name, smiles, description, series)
              VALUES(?,?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Body['name'], Body['chem_name'], Body['smiles'], Body['description'], Body['series'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert body into bodies table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_headgroup(connection, Headgroup, commit=False):
    sql = ''' INSERT INTO headgroups(name, chem_name, smiles, description)
              VALUES(?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Headgroup['name'], Headgroup['chem_name'], Headgroup['smiles'], Headgroup['description'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert headgroup into headgroups table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_linker(connection, Linker, commit=False):
    sql = ''' INSERT INTO linkers(name, chem_name, smiles, description)
              VALUES(?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Linker['name'], Linker['chem_name'], Linker['smiles'], Linker['description'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert linker into linkers table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_metal(connection, Metal, commit=False):
    sql = ''' INSERT INTO metals(name, description)
              VALUES(?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Metal['name'], Metal['description'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert metal into metals table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_halide(connection, Halide, commit=False):
    sql = ''' INSERT INTO halides(name, description)
              VALUES(?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Halide['name'], Halide['description'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert halide into halides table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_ligand(connection, Ligand, commit=False):
    sql = ''' INSERT INTO ligands(body_id,head_id,linker_id,name,smiles,description,series,volume,comments)
              VALUES(?,?,?,?,?,?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Ligand['body_id'], Ligand['head_id'], Ligand['linker_id'], Ligand['name'],Ligand['smiles'],\
                        Ligand['description'],Ligand['series'],Ligand['volume'],Ligand['comments']) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert ligand into ligands table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_perovskite_A(connection, Perovskite, commit=False):
    sql = ''' INSERT INTO perovskites_A(metal_id,halide_id,ligand_id,perov_geom,name,description,synthesized,path,fit,\
              bond_lqe,bond_lqe_diff,bond_lqe_check,angle_var,angle_var_diff,angle_var_check,formed,disordered,volume_fract,comments)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Perovskite['metal_id'], Perovskite['halide_id'], Perovskite['ligand_id'], Perovskite['perov_geom'], Perovskite['name'],\
                         Perovskite['description'], Perovskite['synthesized'], Perovskite['path'], Perovskite['fit'], Perovskite['bond_lqe'],\
                         Perovskite['bond_lqe_diff'], Perovskite['bond_lqe_check'], Perovskite['angle_var'], Perovskite['angle_var_diff'], \
                         Perovskite['angle_var_check'], Perovskite['formed'], Perovskite['disordered'], Perovskite['volume_frac'], Perovskite['comments'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert perovskite into perovskites_A table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_perovskite_AB(connection, Perovskite, commit=False):
    sql = ''' INSERT INTO perovskites_A(metal_id,halide_id,ligand_A_id,ligand_B_id, perov_geom,name,description,synthesized,path,fit,\
              bond_lqe,bond_lqe_diff,bond_lqe_check,angle_var,angle_var_diff,angle_var_check,formed,disordered,volume_fract,comments)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Perovskite['metal_id'], Perovskite['halide_id'], Perovskite['ligand_A_id'], Perovskite['ligand_B_id'], Perovskite['perov_geom'], Perovskite['name'],\
                         Perovskite['description'], Perovskite['synthesized'], Perovskite['path'], Perovskite['fit'], Perovskite['bond_lqe'],\
                         Perovskite['bond_lqe_diff'], Perovskite['bond_lqe_check'], Perovskite['angle_var'], Perovskite['angle_var_diff'], \
                         Perovskite['angle_var_check'], Perovskite['formed'], Perovskite['disordered'], Perovskite['volume_frac'], Perovskite['comments'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert perovskite into perovskites_AB table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def add_fingerprint(connection, Fingerprint, commit=False):
    sql = ''' INSERT INTO fingerprints(ligand_id,type,radius,N_bits,data)
              VALUES(?,?,?,?,?) '''

    try:    
        c = connection.cursor()
        c.execute(sql, ( Fingerprint['ligand_id'], Fingerprint['type'], Fingerprint['radius'], Fingerprint['N_bits'], Fingerprint['data'] ) )
    
    except sqlite3.Error as error:
        print(("\nERROR: Failed to insert perovskite into fingerprints table.\n       Error: {}\n       SQL statement: {}\n".format(error, sql)))
        return -1
    
    if commit: connection.commit()
    return c.lastrowid

def delete_ligand(connection, task_id):
    try:
        c = connection.cursor()
        c.execute('DELETE FROM ligands WHERE id={}'.format(task_id))
        connection.commit()
    except sqlite3.Error as error:
        print(("\nERROR: Failed to delete ligand from ligands table.\n       Error: {}\n".format(error)))
        return False
    
    return True

def retrieve_rows(connection, sql, values=None):
    try:
        c = connection.cursor()
        
        if values == None:
            c.execute(sql)
        else:
            c.execute(sql, values)
        rows = c.fetchall()
    except sqlite3.Error as error:
        print(("\nERROR: Failed to retreive rows from sqlite table.\n       Error: {}\n       SQL statement: '{}'\n".format(error, sql)))
        return []
    
    return rows

# Checks whether or not a record exists in perovskite_A 
# Can specify either the record's ID (to check directly) or the attributes defining that perovskite (since there may be multiple entries sharing the same ligand body but with different perovskites)
# record is a dictionary containing the metal, halide, ligand IDs and perovskite geometry to check
# Returns True or False
def check_perovskite_A(connection, record={}, ID=None):
    if ID != None:
        # c.execute("SELECT count(*) FROM perovskites_A WHERE name = ?", (name,))
        sql = 'SELECT count(1) FROM perovskites_A WHERE id={}'.format(ID)
    else:
        # Check for the presence of all keys in record
        columns = ['metal_id', 'halide_id', 'ligand_id', 'perov_geom']
        error = False
        for key in columns:
            if key not in list(record.keys()):
                error = True
                print('ERROR: functions_db.py::check_perovskite_A(): Missing "{}" in record dictionary. Aborting...'.format(key))
        if error: exit()
        
        sql = 'SELECT count(1) FROM perovskites_A WHERE metal_id={} AND halide_id={} AND ligand_id={} AND perov_geom="{}"'.format(record['metal_id'], record['halide_id'], record['ligand_id'], record['perov_geom'])

    try:
        c = connection.cursor()
        c.execute(sql)
        count = c.fetchone()[0]
        
    except sqlite3.Error as error:
        print(("\nERROR: functions_db.py::perovskite_A_exists(): Failed to count rows in perovskites_A table.\n       Error: {}\n".format(error)))
        return False
    
    return False if count == 0 else True

# Checks whether or not a record exists in bodies
# Can specify either the record's ID (to check directly) or the attributes defining that body (name, series)
# record is a dictionary containing the attributes (name, series) to check
# Returns True or False
def check_body(connection, record={}, ID=None):
    if ID != None:
        # c.execute("SELECT count(*) FROM perovskites_A WHERE name = ?", (name,))
        sql = 'SELECT count(1) FROM bodies WHERE id={}'.format(ID)
    else:
        # Check for the presence of all keys in record
        columns = ['name', 'series']
        error = False
        for key in columns:
            if key not in list(record.keys()):
                error = True
                print('ERROR: functions_db.py::check_body(): Missing "{}" in record dictionary. Aborting...'.format(key))
        if error: exit()
        
        sql = 'SELECT count(1) FROM bodies WHERE name="{}" AND series="{}"'.format(record['name'], record['series'])

    try:
        c = connection.cursor()
        c.execute(sql)
        count = c.fetchone()[0]
        
    except sqlite3.Error as error:
        print(("\nERROR: functions_db.py::check_body(): Failed to count rows in bodies table.\n       Error: {}\n".format(error)))
        return False
    
    return False if count == 0 else True

# Checks whether or not a record exists in ligands
# Can specify either the record's ID (to check directly) or the attributes defining that body (body_id, head_id, linker_id, name, series)
# record is a dictionary containing the attributes (body_id, head_id, linker_id, name, series) to check
# Returns True or False
def check_ligand(connection, record={}, ID=None):
    if ID != None:
        # c.execute("SELECT count(*) FROM perovskites_A WHERE name = ?", (name,))
        sql = 'SELECT count(1) FROM ligands WHERE id={}'.format(ID)
    else:
        # Check for the presence of all keys in record
        columns = ['body_id', 'head_id', 'linker_id', 'name', 'series']
        error = False
        for key in columns:
            if key not in list(record.keys()):
                error = True
                print('ERROR: functions_db.py::check_ligand(): Missing "{}" in record dictionary. Aborting...'.format(key))
        if error: exit()
        
        sql = 'SELECT count(1) FROM ligands WHERE body_id={} AND head_id={} AND linker_id={} AND name="{}" AND series="{}"'.format(record['body_id'], record['head_id'], record['linker_id'], record['name'], record['series'])

    try:
        c = connection.cursor()
        c.execute(sql)
        count = c.fetchone()[0]
        
    except sqlite3.Error as error:
        print(("\nERROR: functions_db.py::check_ligand(): Failed to count rows in ligands table.\n       Error: {}\n".format(error)))
        return False
    
    return False if count == 0 else True

# Checks whether or not a record exists in fingerprints
# Can specify either the record's ID (to check directly) or the attributes defining that body (body_id, head_id, linker_id, name, series)
# record is a dictionary containing the attributes (body_id, head_id, linker_id, name, series) to check
# Returns True or False
def check_fingerprint(connection, record={}, ID=None):
    if ID != None:
        sql = 'SELECT count(1) FROM fingerprints WHERE id={}'.format(ID)
    else:
        # Check for the presence of all keys in record
        columns = ['ligand_id', 'type', 'radius', 'N_bits']
        error = False
        for key in columns:
            if key not in list(record.keys()):
                error = True
                print('ERROR: functions_db.py::check_fingerprint(): Missing "{}" in record dictionary. Aborting...'.format(key))
        if error: exit()
        
        sql = 'SELECT count(1) FROM fingerprints WHERE ligand_id={} AND type="{}" AND radius={} AND N_bits={}'.format(record['ligand_id'], record['type'], record['radius'], record['N_bits'])

    try:
        c = connection.cursor()
        c.execute(sql)
        count = c.fetchone()[0]
        
    except sqlite3.Error as error:
        print(("\nERROR: functions_db.py::check_fingerprint(): Failed to count rows in fingerprints table.\n       Error: {}\n".format(error)))
        return False
    
    return False if count == 0 else True


# Return a dictionary of halides where the key is the name and the value is its db index
def get_indices(connection, table=None):
    
    if table is None or table not in ['bodies', 'headgroups', 'linkers', 'metals', 'halides', 'ligands']:
        print('functions.db::get_indices(connection, table): table is not recognized. Accepts only bodies, headgroups, linkers, metals, halides, ligands.\nAborting....\n')
        exit()
    
    try:
        c = connection.cursor()
        c.execute('SELECT id,name FROM {}'.format(table))
        rows = c.fetchall()
    except sqlite3.Error as error:
        print(("\nERROR: Failed to retreive rows from {} table.\n       Error: {}\n".format(table, error)))
        return {}
    
    Indices = {}
    for r in rows:
        Indices[r[1]] = r[0]
        
    return Indices
    
def main():
    print('Functions_db::main()')
    return

if __name__ == '__main__':
    main()