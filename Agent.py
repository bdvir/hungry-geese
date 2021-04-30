def agent(obs_dict, config_dict):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    # print("Config dict", config_dict) # Config is static
    # print("Observation dict", obs_dict)  # This changes over each timestep (your observations change over time poggers)
    # print(obs_dict['step'])  # This and obs_dict.step are the exact same. Anybody know why?

    observation = Observation(obs_dict)  # -> Why is obs_dict wrapped in Observation? What does that do?
    # print('ho', observation)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    food = observation.food[0]
    food_row, food_column = row_col(food, configuration.columns)

    if food_row > player_row:
        return Action.SOUTH.name
    if food_row < player_row:
        return Action.NORTH.name
    if food_column > player_column:
        return Action.EAST.name
    return Action.WEST.name