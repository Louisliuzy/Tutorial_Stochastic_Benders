#!/usr/bin/env python3
"""
-------------------
MIT License

Copyright (c) 2024  Zeyu Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-------------------
Description:
    Two-stage stochastic program,
    Solved with the L-shaped method
-------------------
"""

# import
import logging
import numpy as np
import gurobipy as grb


class Two_Stage_Stochastic_Program():
    """
    Two_Stage_Stochastic_Program
    """

    def __init__(self, name):
        """
        init
        """
        super().__init__()
        self.name = name
        # problem data
        # ++++++++++++++++++++++++++++++++
        self.product = ["A", "B"]
        self.resource = ["alpha", "beta", "gamma"]
        # production
        self.m = {
            "A": {"alpha": 1, "beta": 3, "gamma": 0},
            "B": {"alpha": 1, "beta": 4, "gamma": 1}
        }
        # profit
        self.q = {
            "A": 40, "B": 50
        }
        # cost
        self.c = {
            "alpha": 10, "beta": 1, "gamma": 1
        }
        # salvage
        self.s = {
            "alpha": 0, "beta": 0.1, "gamma": 0.1
        }
        # scenarios
        self.scenario = [1, 2, 3]
        # probability
        self.p = {
            1: 0.3, 2: 0.5, 3: 0.2
        }
        self.d = {
            1: {'A': 10, 'B': 30},
            2: {'A': 20, 'B': 60},
            3: {'A': 40, 'B': 80}
        }
        # ================================
        # build master problem
        self.MP = self.__build_MP()
        # record first stage variables
        # ++++++++++++++++++++++++++++++++
        # x
        self.var_x = {
            j: self.MP.getVarByName("x_{}".format(j))
            for j in self.resource
        }
        # ================================
        # always record theta
        self.var_theta = self.MP.getVarByName("theta")
        # subproblems, undefined for now
        self.SP = {}

    def __build_MP(self):
        """
        build the master problem here
        """
        # start model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 0)
        model.setParam("IntFeasTol", 1e-9)
        # model.setParam("MIPFocus", 3)
        # model.setParam("NumericFocus", 3)
        model.setParam("Presolve", -1)
        # ++++++++++++++++++++++++++++++++
        # variables
        # x
        var_x = {}
        for j in self.resource:
            var_x[j] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="x_{}".format(j)
            )
        model.update()
        # theta
        var_theta = model.addVar(
            lb=-1e8, ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name="theta"
        )
        model.update()
        # objective
        obj = grb.quicksum([
            grb.quicksum([
                self.c[j] * var_x[j]
                for j in self.resource
            ]),
            var_theta
        ])
        model.setObjective(obj, grb.GRB.MINIMIZE)
        # constraints
        # ================================
        return model

    def __build_dual(self, k):
        """
        build dual of the subproblem at scenario k
        """
        # start model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 0)
        model.setParam("IntFeasTol", 1e-9)
        # model.setParam("MIPFocus", 3)
        # model.setParam("NumericFocus", 3)
        model.setParam("InfUnbdInfo", 1)
        model.setParam("Presolve", -1)
        # variables
        # --------------------------------
        # lambda
        var_lambda = {}
        for j in self.resource:
            var_lambda[j] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="lambda_{}".format(j)
            )
        model.update()
        # mu
        var_mu = {}
        for i in self.product:
            var_mu[i] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=0,
                vtype=grb.GRB.CONTINUOUS,
                name="mu_{}".format(i)
            )
        model.update()
        # objective
        # --------------------------------
        obj = grb.quicksum([
            grb.quicksum([
                self.var_x[j].X * var_lambda[j]
                for j in self.resource
            ]),
            grb.quicksum([
                self.d[k][i] * var_mu[i]
                for i in self.product
            ])
        ])
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        # constraints
        # --------------------------------
        # first one
        for i in self.product:
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        self.m[i][j] * var_lambda[j]
                        for j in self.resource
                    ]),
                    var_mu[i]
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=-self.q[i]
            )
        model.update()
        # second one
        for j in self.resource:
            model.addLConstr(
                lhs=var_lambda[j],
                sense=grb.GRB.LESS_EQUAL,
                rhs=-self.s[j]
            )
        model.update()
        # --------------------------------
        return model

    def L_shaped(self, write_log=True):
        """
        The L-shaped algorithm
        """
        # logging
        logging.basicConfig(
            filename='{}_L-shaped.log'.format(self.name), filemode='w+',
            format='%(levelname)s - %(message)s', level=logging.INFO
        )
        # starting iteration
        iter = 0
        while True:
            # logging
            if write_log:
                logging.info("Iteration: {}".format(iter))
            # solve MP
            self.MP.optimize()
            # solutions
            if self.MP.status != grb.GRB.OPTIMAL:
                raise ValueError(
                    "First stage optimality code {}".format(self.MP.status)
                )
            # variables to use
            SP_feasible = True
            # ++++++++++++++++++++++++++++++++
            lambda_val, mu_val = {}, {}
            # ================================
            # iterating through all scenarios
            for k in self.scenario:
                # build dual
                self.SP[k] = self.__build_dual(k)
                self.SP[k].optimize()
                # check status
                if self.SP[k].status == grb.GRB.OPTIMAL:
                    # record solutions
                    # ++++++++++++++++++++++++++++++++
                    # lambda
                    for j in self.resource:
                        lambda_val[k, j] = self.SP[k].getVarByName(
                            "lambda_{}".format(j)
                        ).X
                    # mu
                    for i in self.product:
                        mu_val[k, i] = self.SP[k].getVarByName(
                            "mu_{}".format(i)
                        ).X
                    # ================================
                elif self.SP[k].status == grb.GRB.UNBOUNDED:
                    if write_log:
                        logging.info(
                            "    Adding a feasibility cut "
                            "for scenario {} ...".format(k)
                        )
                    SP_feasible = False
                    # record unbounded rays
                    # --------------------------------
                    # lambda
                    lambda_ray = {
                        j: self.SP[k].getVarByName(
                            "lambda_{}".format(j)
                        ).UnbdRay
                        for j in self.resource
                    }
                    # mu
                    mu_ray = {
                        i: self.SP[k].getVarByName(
                            "mu_{}".format(i)
                        ).UnbdRay
                        for i in self.product
                    }
                    # add feasibility cut
                    self.MP.addLConstr(
                        # fill in
                    )
                    self.MP.update()
                    # ================================
                    break
                else:
                    raise ValueError(
                        "Second stage at {} optimality code {}.".format(
                            k, self.SP[k].status
                        )
                    )
            # if all subproblems are feasible
            if SP_feasible:
                # calculate expected value
                value = np.sum([
                    self.p[k] * self.SP[k].ObjVal
                    for k in self.scenario
                ])
                if write_log:
                    logging.info("    All subproblems feasible.")
                    logging.info("    Value: {}; theta: {}".format(
                        value, self.var_theta.X
                    ))
                # checking optimality condition
                epsilon = 1e-6
                # if optimal
                if value <= self.var_theta.X + epsilon:
                    if write_log:
                        logging.info("Optimal!")
                        logging.info(
                            "MP objective: {}".format(self.MP.objVal)
                        )
                        logging.info("SP objective:")
                        for k in self.scenario:
                            logging.info("    Scenario {}: {}".format(
                                k, self.SP[k].ObjVal
                            ))
                    break
                # not optimal
                else:
                    if write_log:
                        logging.info("    Adding an optimality cut...")
                    # add optimality cut
                    # --------------------------------
                    self.MP.addLConstr(
                        # fill in
                    )
                    self.MP.update()
                    # ================================
            # any scenario infeasible
            else:
                pass
            iter += 1
        return 0

    def build_extensive_form(self):
        """
        extensive form
        """
        # start model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 0)
        model.setParam("IntFeasTol", 1e-9)
        # model.setParam("MIPFocus", 3)
        # model.setParam("NumericFocus", 3)
        model.setParam("Presolve", -1)
        # variables
        # x
        var_x = {}
        for j in self.resource:
            var_x[j] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="x_{}".format(j)
            )
        model.update()
        # y
        var_y = {}
        for k in self.scenario:
            for i in self.product:
                var_y[k, i] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="y_{}_{}".format(k, i)
                )
        model.update()
        # z
        var_z = {}
        for k in self.scenario:
            for j in self.resource:
                var_z[k, j] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="z_{}_{}".format(k, j)
                )
        model.update()
        # objective
        obj = grb.quicksum([
            grb.quicksum([
                self.c[j] * var_x[j]
                for j in self.resource
            ]),
            grb.quicksum([
                self.p[k] * grb.quicksum([
                    grb.quicksum([
                        -self.q[i] * var_y[k, i]
                        for i in self.product
                    ]),
                    grb.quicksum([
                        -self.s[j] * var_z[k, j]
                        for j in self.resource
                    ]),
                ])
                for k in self.scenario
            ]),
        ])
        model.setObjective(obj, grb.GRB.MINIMIZE)
        # constriants
        for k in self.scenario:
            # resource balance
            for j in self.resource:
                model.addLConstr(
                    lhs=var_z[k, j],
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum([
                        var_x[j],
                        grb.quicksum([
                            -self.m[i][j] * var_y[k, i]
                            for i in self.product
                        ])
                    ])
                )
            # demand bound
            for i in self.product:
                model.addLConstr(
                    lhs=var_y[k, i],
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=self.d[k][i]
                )
        model.update()
        return model
