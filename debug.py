import matplotlib.pyplot as plt


def draw(rectangle):
    rect = plt.Rectangle((0.1, 0.1), 0.5, 0.3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, hyper in enumerate(rectangle):
        xrange = hyper.hyperrange[0]

        yrange = hyper.hyperrange[1]
        site = ((xrange[0], yrange[0]))
        y = yrange[1]-yrange[0]
        x = xrange[1]-xrange[0]
        clos = ['r', 'g', 'blue', 'yellow']
        hatchs = ['/', '\\', '|', '-']
        rect = plt.Rectangle(site, x, y,
                             alpha=0.1, fill=False, lw=1, edgecolor='black')  # ,fc = clos[i%4]hatch=hatchs[i%4]
        ax.add_patch(rect)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.savefig("-1.png", dpi=400)


def draw_tree(hist, num=0, tree_dict=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    children_list = []
   
    tratree(hist, ax, children_list,tree_dict)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.savefig(f"{num}.png", dpi=200)
    print(len(children_list))


def tratree(tree, ax, children_list,tree_dict):
    children_list.append(tree)
    clos = ['r', 'g', 'blue', 'yellow']
    hatchs = ['/', '\\', '|', '-']
    site = tuple(tree.mins)
    high = tree.maxs[0]-tree.mins[0]
    width = tree.maxs[1]-tree.mins[1]
    rect = plt.Rectangle(site, high, width,
                         alpha=0.1, fill=True, lw=1, edgecolor='black', fc=clos[len(children_list) % 4])
    ax.add_patch(rect)
    for cid in tree.children:
        child = tree_dict[cid]
        tratree(child, ax, children_list,tree_dict)


def check_node(plan, nodetype):
    head = plan['Node Type']
    """ if head == 'Hash':
        print(plan) """
    nodetype.add(head)
    try:
        for child in plan['Plans']:
            check_node(child, nodetype)
    except:
        pass


def check_type(plan, type, nodes, query=None):
    head = plan['Node Type']
    loops = plan['Actual Loops']
    # if  para and loops!=1:

    if head == type and loops == 1:
        # if plan['Index Cond'][-2].isnumeric():
        print(str(plan))
        if query is not None:
            print(query)
        nodes.append(plan)
    try:
        for child in plan['Plans']:
            check_type(child, type, nodes, query)
    except:
        pass
