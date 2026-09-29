from .ocp_centroidal_vel import OCPCentroidalVel
from .ocp_centroidal_vel_qp import OCPCentroidalVelQP
from .ocp_centroidal_acc import OCPCentroidalAcc
from .ocp_whole_body_acc import OCPWholeBodyAcc
from .ocp_whole_body_aba import OCPWholeBodyABA
from .ocp_whole_body_rnea import OCPWholeBodyRNEA
from .ocp_whole_body_rnea_qp import OCPWholeBodyRNEAQP


def make_ocp(dynamics, dyn_args, **kwargs):
    ocp_classes = {
        "centroidal_vel": OCPCentroidalVel,
        "centroidal_vel_qp": OCPCentroidalVelQP,
        "centroidal_acc": OCPCentroidalAcc,
        "whole_body_acc": OCPWholeBodyAcc,
        "whole_body_aba": OCPWholeBodyABA,
        "whole_body_rnea": OCPWholeBodyRNEA,
        "whole_body_rnea_qp": OCPWholeBodyRNEAQP,
    }

    if dynamics not in ocp_classes:
        raise ValueError(f"Unknown dynamics type: {dynamics}")

    args = dyn_args.copy()
    args.update(kwargs)

    ocp = ocp_classes[dynamics](**args)
    ocp.setup_problem()
    ocp.set_weights()

    return ocp
