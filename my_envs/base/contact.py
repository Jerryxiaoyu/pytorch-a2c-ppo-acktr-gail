
def get_contact_info(env, verbose = False ):
    d = env.unwrapped.data

    #print(d.ncon)
    geom_list =[]
    for coni in range(d.ncon):
        con = d.contact[coni]
        if con.geom1 == 0:
            geom_list.append(con.geom2)
        if verbose:
            print('  Contact %d:' % (coni,))
            print('    dist     = %0.3f' % (con.dist,))
            print('    pos      = %s' % (str_mj_arr(con.pos),))
            print('    frame    = %s' % (str_mj_arr(con.frame),))
            print('    friction = %s' % (str_mj_arr(con.friction),))
            print('    dim      = %d' % (con.dim,))
            print('    geom1    = %d' % (con.geom1,))
            print('    geom2    = %d' % (con.geom2,))

    return geom_list