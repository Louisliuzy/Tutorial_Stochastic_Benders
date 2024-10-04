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
    Main file
    Solving the production problem with L-shaped method
-------------------
"""


# import
from stochastic_programing import Two_Stage_Stochastic_Program


# main
def main():
    """
    main
    """
    # define problem
    production = Two_Stage_Stochastic_Program(
        name="production"
    )
    # build extensive form
    extensive = production.build_extensive_form()
    # solve entensive
    extensive.optimize()
    # solve two-stage
    production.L_shaped()
    # compare objective
    print("Extensive: ", extensive.ObjVal)
    print("Two-stage: ", production.MP.ObjVal)
    return


if __name__ == "__main__":
    main()
