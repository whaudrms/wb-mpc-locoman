# Arguments for each dynamics model
DYN_ARGS = {
    "centroidal_vel": {
        "include_base": True,  # whether base velocity is part of the input
    },
    "centroidal_vel_qp": {
        "include_base": True,  # explicit full velocity; False eliminates base velocity
    },
    "centroidal_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_rnea": {
        "include_acc": True,  # whether to include accelerations in the input (necessary for Fatrop due to structure detection!)
    },
    "whole_body_rnea_qp": {
        "include_acc": True,  # accelerations and torques at every horizon node
    },
    "whole_body_aba": {}  # the input just contains joint torques
}

# Arguments for each solver
SOLVER_ARGS = {
    "qp": {
        # Default backend: centroidal_vel_qp -> qpoases; whole_body_rnea_qp -> osqp.
        # main.py selects the backend and condensing; "qp" retains model defaults.
        "qpoases_opts": {
            "printLevel": "none",
            "nWSR": 10000,  # Active-set recalculations within ONE QP solve.
            "terminationTolerance": 1e-8,
            "initialStatusBounds": "inactive",
            "enableFullLITests": True,  # Robust handling of dependent constraints.
            "enableCholeskyRefactorisation": 1,  # Re-factor as the active set changes.
            "numRefinementSteps": 3,  # Reduce hot-start roundoff in late horizon QPs.

        },
        "hpipm_mode": "robust",
        "hpipm_root": None,  # HPIPM_ROOT env, project .deps/hpipm, or system installation.
        "hpipm_opts": {
            "iter_max": 100,
            "tol_stat": 1e-7,
            "tol_eq": 1e-8,
            "tol_ineq": 1e-8,
            "tol_comp": 1e-8,
            "reg_prim": 1e-12,
            "warm_start": 0,  # MPC reference is shifted; QP coordinates can change.
        },
        "regularization": 1e-8,
        "max_qp_violation": 1e-3,
        "opts": {  # OSQP options only.
            "verbose": False,
            "max_iter": 100000,  # OSQP iterations within ONE QP solve
            "eps_abs": 1e-6,
            "eps_rel": 1e-7,
            "polish": True,
            "warm_start": True,
        },
    },
    "fatrop": {
        "opts": {
            "print_time": False,  # Keep native timing tables out of the measured solve.
            "expand": True,
            "structure_detection": "auto",
            "debug": True,
            "fatrop.print_level": 0,
            "fatrop.max_iter": 10,
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 1e-4,
            "fatrop.warm_start_init_point": True,
            "fatrop.warm_start_mult_bound_push": 1e-7,
            "fatrop.bound_push": 1e-7,
        }
    },
    "ipopt": {
        "opts": {
            "print_time": False,
            "ipopt.sb": "yes",  # Suppress the solver banner during timed calls.
            "expand": True,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-3,
            "ipopt.mu_init": 1e-4,
            # "ipopt.warm_start_init_point": True,
            # "ipopt.warm_start_mult_bound_push": 1e-7,
            # "ipopt.bound_push": 1e-7,
        }
    },
    "osqp": {
        "iters": 2,  # number of SQP iterations
        "opts": {
            "verbose": False,
            "max_iter": 20,  # number of sub-iterations for each QP
            "alpha": 1.4,
            "rho": 2e-2,
            "warm_start": True,
            "adaptive_rho": False,
        }
    }
}
