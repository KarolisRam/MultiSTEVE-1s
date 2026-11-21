def place_tree_commands(x, y, z, tree_type='oak', relative=True, villager=True, villager_profession='unemployed',
                        villager_type='plains', logs_above_villager=False):
    """
    Makes a list of creative commands to execute to place an oak or birch tree with a villager instead of the trunk.
    :param x: x coordinate for the tree
    :param y: y coordinate for the tree
    :param z: z coordinate for the tree
    :param tree_type: 'oak' or 'birch', others have different shapes
    :param relative: use absolute or relative to the player coordinates
    :param villager: whether to replace bottom two logs with a villager and place barrier blocks around
    :param villager_profession: see https://minecraft.fandom.com/wiki/Villager#Professions for a full grid of pictures
    of profession and type combinations. Professions: unemployed, nitwit, armorer, butcher, cartographer, cleric,
    farmer, fisherman, fletcher, leatherworker, mason, shepherd, toolsmith, weaponsmith
    :param villager_type: desert, jungle, plains, savanna, snow, swamp, taiga
    :param logs_above_villager: keep the log blocks above the villager instead of clearing them with air
    :return: list of creative commands to execute
    """
    t = '~' if relative else ''
    log = f'{tree_type}_log'
    leaves = f'{tree_type}_leaves'
    villager_data = f'{{VillagerData:{{profession:{villager_profession},type:{villager_type}}}}}'
    fills = [
        [0, 0, 0, 0, 4, 0, log],
        [-2, 2, -2, 2, 3, 2, leaves],
        [0, 4, -1, 0, 5, 1, leaves],
        [-1, 4, 0, 1, 5, 0, leaves],
        [0, 0, 0, 0, 4, 0, log],
    ]
    blocks = [
        [-2, 3, -2, 'air'],
        [2, 3, -2, 'air'],
        [2, 2, -2, 'air'],
    ]
    blocks += [
        # [0, -1, 0, 'diamond_ore'],
    ]
    summons = []
    if villager:
        blocks += [
            [0, 0, 0, 'air'],
            [0, 1, 0, 'air'],
            [-1, 0, 0, 'barrier'],
            [1, 0, 0, 'barrier'],
            [0, 0, -1, 'barrier'],
            # [0, 0, -1, 'oak_trapdoor[facing=north,half=bottom,open=true]'],
            [0, 0, 1, 'barrier'],
        ]
        if not logs_above_villager:
            blocks += [
                [0, 2, 0, 'air'],
                [0, 3, 0, 'air'],
            ]
        summons += [
            [0, 0, 0, 'villager'],
        ]

    commands = []
    for fill in fills:
        x1, y1, z1, x2, y2, z2, obj = fill
        commands.append(f'/fill {t}{x+x1} {t}{y+y1} {t}{z+z1} {t}{x+x2} {t}{y+y2} {t}{z+z2} minecraft:{obj}')
    for block in blocks:
        x1, y1, z1, obj = block
        commands.append(f'/setblock {t}{x+x1} {t}{y+y1} {t}{z+z1} minecraft:{obj}')
    for summon in summons:
        x1, y1, z1, obj = summon
        commands.append(f'/summon {obj} {t}{x+x1} {t}{y+y1} {t}{z+z1} {villager_data}')
    return commands
