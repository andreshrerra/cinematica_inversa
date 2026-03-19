from fastapi import FastAPI
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=4.0, a=1.0, alpha=np.pi/2, offset=np.pi/2),
        rtb.RevoluteDH(d=0.0, a=12.0, alpha=0.0, offset=np.pi/2),
        rtb.RevoluteDH(d=-1.5, a=0.0, alpha=np.pi/2, offset=0.0),
        rtb.RevoluteDH(d=12.0, a=0.0, alpha=-np.pi/2, offset=0.0),
        rtb.RevoluteDH(d=1.0, a=6.5, alpha=0.0, offset=-np.pi/2),
    ],
    name="Robot5DOF"
)

@app.post("/ik")
def calcular_ik(data: dict):
    px = data["px"]
    py = data["py"]
    pz = data["pz"]
    R = data["R"]
    P = data["P"]
    Y = data["Y"]

    # Transformación objetivo
    T_objetivo = (
        sm.SE3.Trans(px, py, pz) *
        sm.SE3.RPY([R, P, Y], unit='deg', order='xyz')
    )

    sol = robot.ikine_LM(
        T_objetivo,
        mask=[1,1,1,1,1,1],
        q0=np.zeros(5),
        tol=1e-6
    )

    if not sol.success:
        return {"error": sol.reason}

    q = np.rad2deg(sol.q)

    return {
        "q1": float(q[0]),
        "q2": float(q[1]),
        "q3": float(q[2]),
        "q4": float(q[3]),
        "q5": float(q[4])
    }   
