import pandas as pd
import numpy as np


def read_max_eff():
    # 본 예제에서, df의 index는 torque, columns는 speed
    # index_col에 값을 주어 특정 열을 index로 사용
    df_id = pd.read_csv(
        "max_eff_id.csv", dtype=np.float, index_col=0, delimiter=","
    )
    df_iq = pd.read_csv(
        "max_eff_iq.csv", dtype=np.float, index_col=0, delimiter=","
    )

    torque_index = df_id.index  # index: 행 방향 이름 (torque) 리스트 반환

    input_arr = []
    output_arr = []

    for torque in torque_index:
        # index가 torque 값인 행을 하나씩 뽑는다. (Series 변수형)
        id_row = df_id.loc[torque, :]
        iq_row = df_iq.loc[torque, :]

        # 전류 값이 없는 칸(nan)을 False로,
        # 전류 값이 있는 칸을 True로 표시하는 리스트
        not_nan_list = ~np.isnan(id_row.values)

        # 데이터셋의 입력 변수 준비
        speed_list = id_row.loc[not_nan_list].index.astype(
            "float"
        )  # (예) [1000, 2000, 3000, ...]
        torque_list = torque * np.ones(
            np.size(speed_list)
        )  # speed_list와 같은 크기의 torque_list 생성  (예) [5, 5, 5, ...]

        # 두 개의 행 벡터를 열로 바꾸고, 열을 늘리는 방향으로 append
        new_input = np.c_[torque_list, speed_list]
        new_output = np.c_[
            id_row.loc[not_nan_list].values, iq_row.loc[not_nan_list].values
        ]

        try:  # 두 번째 torque 이후 (5~50)
            # 두 행렬을 행을 늘리는 방향으로 append
            input_arr = np.r_[input_arr, new_input]
            output_arr = np.r_[output_arr, new_output]
        except ValueError:  # 첫 번째 torque (0)
            input_arr = new_input
            output_arr = new_output

    return input_arr, output_arr
