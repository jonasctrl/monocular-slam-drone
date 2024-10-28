
from __future__ import annotations
import random
import math
import numpy as np
from typing import Optional, Dict, List, Tuple

STRA_COST = 1
DIAG_COST = math.sqrt(2)

BOTH_END = -1
START_END = 0
GOAL_END = 1

# random.seed(15)
# random.seed(16)
random.seed(17)

# DEBUG=1
DEBUG=0

def dbg(*args, **kwargs):
    if DEBUG != 0:
        print(*args, **kwargs)

def iend(end):
    return START_END if end == GOAL_END else GOAL_END

class Node:
    def __init__(self, pos):
        self.pos: tuple = (int(pos[0]), int(pos[1]))

        self.is_root:bool = False
        self.parent: Optional['Node'] = None
        self.cost:float = float("inf")
        self.children: Dict[tuple, 'Node'] = dict()

    def clone(self) -> Node:
        new_node = Node(self.pos)
        new_node.cost = self.cost
        return new_node

    def __update_cost(self) -> None:
        eq = 0
        par = self.parent
        if self.is_root:
            self.cost = 0
            return 
        elif par is None:
            self.cost = float("inf")
            return

        for i in range(len(self.pos)):
            if self.pos[i] != par.pos[i]:
                eq += 1

        self.cost = par.cost + math.sqrt(eq)

    def update_costs_rec(self) -> None:
        self.__update_cost()

        for c in self.children.values():
            c.update_costs_rec()


    def validate_relationships(self) -> None:
        if self.parent is None:
            return 
        
        assert self.parent.children.get(self.pos) == self

    def connect_child(self, child:Node) -> None:
        assert child
        child.set_root()
        child.parent = self
        self.children[child.pos] = child

        child.update_costs_rec()

    def breakoff(self) -> None:
        parent = self.parent
        if parent is None:
            return 
        
        self.parent = None
        del parent.children[self.pos]

        self.update_costs_rec()

    def delete(self):
        for c in self.children.values():
            c.parent = None
            c.update_costs_rec()

        self.children.clear()
        self.breakoff()
    

    def set_root(self) -> None:
        """Reassign parent-child relationships to make 'node' the new root of the tree."""
        current = self
        previous_parent = None
        
        while current is not None:
            next_parent = current.parent            
            if next_parent is not None:
                del next_parent.children[current.pos]
                next_parent.is_root = False
                current.children[next_parent.pos] = next_parent
                
            current.parent = previous_parent
            previous_parent = current
            current = next_parent

    def dist_to(self, node:Node) -> float:
        ret = np.linalg.norm(np.array(node.pos) - np.array(self.pos))
        return float(ret)

class DRRT:
    def __init__(self, start:tuple, goal:tuple, shape:tuple, step_size=1, max_iter=1000, goal_sample_rate=0.05):
        self.start: Node = Node(start)
        self.start.is_root = True
        self.start.cost = 0

        self.goal: Node = Node(goal)
        self.goal.is_root = True
        self.goal.cost = 0

        self.main_bridge: Optional[Node] = None

        self.total_cost = float("inf")
        self.shp: Tuple[int, int] = shape
        self.step_size: int = step_size
        self.max_iter: int = max_iter
        self.goal_sample_rate: float = goal_sample_rate

        self.obst_dict: Dict[tuple,int ]= dict()
        self.bridge_dict: Dict[tuple,'Node'] = dict()
        self.forward_dict: Dict[tuple,'Node'] = dict()
        self.bacward_dict: Dict[tuple,'Node'] = dict()
        self.forward_dict[self.start.pos] = self.start
        self.bacward_dict[self.goal.pos] = self.goal

        self.node_dict: List[Dict[tuple,'Node']] = [self.forward_dict, self.bacward_dict]

    def __update_main_bridge(self) -> None:
        if self.main_bridge is None:
            br_list = list(self.bridge_dict.values())
            if len(br_list) > 0:
                dbg(f"updating main_bridge {'None' if self.main_bridge is None else self.main_bridge.pos} => {br_list[0].pos}")
                self.main_bridge = br_list[0]
                self.__update_total_cost()
                
        elif self.bridge_dict.get(self.main_bridge.pos) is None:
                br_list = list(self.bridge_dict.values())
                if len(br_list) > 0:
                    self.main_bridge = br_list[0]
                else:
                    self.main_bridge = None

                self.__update_total_cost()

    def __delete_node(self, end:int, pos:tuple) -> None:
        if end == BOTH_END:
            self.__delete_node(START_END, pos)
            self.__delete_node(GOAL_END, pos)

        node = self.node_dict[end].get(pos)
        if node is not None:
            dbg(f"deleting node:{pos}")
            node.delete()
            del self.node_dict[end][node.pos]


        if self.bridge_dict.get(pos) is not None:
            del self.bridge_dict[pos]
            self.__update_main_bridge()

    def __add_node(self, parent:Optional[Node], child:Node, end:int) -> bool:
        if self.node_dict[end].get(child.pos) is not None:
            return False
        if self.obst_dict.get(child.pos) is not None:
            return False
        else:
            dbg(f"new[{end}]\t{'None' if parent is None else parent.pos} => {child.pos}")
            if parent is not None:
                parent.connect_child(child)
            self.node_dict[end][child.pos] = child

            if self.node_dict[iend(end)].get(child.pos) is not None:
                self.bridge_dict[child.pos] = child.clone()
                self.__update_main_bridge()

            return True
        
    def __delete_orphans(self):
        for e in list(self.forward_dict.values()):
            if e.cost == float("inf"):
                self.__delete_node(START_END, e.pos)

        for e in list(self.bacward_dict.values()):
            if e.cost == float("inf"):
                self.__delete_node(GOAL_END, e.pos)

    def __set_new_root(self, pos:tuple, end:int) -> bool:
        new_node = self.node_dict[end].get(pos)
        if new_node is None:
            # No node in new position. Thus, creating new one
            new_node = Node(pos)
            if not self.__add_node(None, new_node, end):
                return False
            
        new_node.set_root()

        if end == START_END:
            dbg(f"old start {self.start.pos} {self.start}")
            self.start.is_root = False
            self.start = new_node
            dbg(f"new start {self.start.pos} {self.start}")
        else:
            dbg(f"old goal {self.goal.pos} {self.goal}")
            self.goal.is_root = False
            self.goal = new_node
            dbg(f"new goal {self.goal.pos} {self.goal}")

        new_node.is_root = True

        self.__update_total_cost()
        self.__delete_orphans()

        return True
    
    def add_obstacle(self, pos:tuple) -> None:
        (x, y, v) = pos
        ob_pos = (x, y)
        if v == -1:
            self.__delete_node(BOTH_END, ob_pos)
            self.obst_dict[ob_pos] = 1
        elif v == 0:
            if self.obst_dict.get(ob_pos) is not None:
                del self.obst_dict[ob_pos]
        else:
            raise Exception("Unsupported obstacle value:{v} should be 0 or -1")

    def update_obstacles(self, ob_list:List[tuple], clear:bool=False) -> None:
        if clear:
            self.obst_dict.clear()

        for o in ob_list:
            (x, y, v) = o
            ob = (x, y)
            if ob != self.start.pos and ob != self.goal.pos:
                self.add_obstacle(o)
        self.__update_total_cost()
        self.__delete_orphans()

    def __get_random_node(self, end:int) -> Node:
        guess = random.random()
        if guess < self.goal_sample_rate:
            pos = self.goal.pos if end == START_END else self.start.pos
            return Node(pos)
        else:
            # Do not select same end "goal"
            while True:
                r_x = random.randint(0, self.shp[0])
                r_y = random.randint(0, self.shp[1])
                pos = self.goal.pos if end == GOAL_END else self.start.pos
                if not (pos == (r_x, r_y) and len(self.node_dict[end]) == 1):
                    break

            return Node((r_x, r_y))

    def __get_nearest_conn_node(self, random_node:Node, end:int) -> Node:
        dists = [(random_node.dist_to(node), node) \
                for node in self.node_dict[end].values() \
                if node.cost != float("inf") and \
                random_node.pos != node.pos]
        dists = sorted(dists, key=lambda x: x[0])
        return dists[0][1]

    def __steer(self, from_node:Node, to_node:Node) -> Node:
        from_pos = np.array(from_node.pos)
        to_pos = np.array(to_node.pos)

        vec = to_pos - from_pos
        distance = max(np.abs(vec))
        sq_off = vec * self.step_size / distance
        pos_off = np.array([0,0])
        for i, p in enumerate(sq_off):
            if p > 0.333:
                pos_off[i] = 1
            elif p < -0.333:
                pos_off[i] = -1

        new_pos = from_pos + pos_off
        new_node = Node(new_pos)
        return new_node

    def __get_path_from_bridge(self, start_node:Node) -> List[Node]:
        path = [start_node]
        node = start_node
        while node.parent is not None:
            node = node.parent
            path.append(node)

        return path

    def ___get_node_path(self, bridge:Node, direction:int) -> List[Node]:
        f_node = self.forward_dict.get(bridge.pos)
        b_node = self.bacward_dict.get(bridge.pos)
        assert f_node is not None
        assert b_node is not None

        f_path = self.__get_path_from_bridge(f_node)
        b_path = self.__get_path_from_bridge(b_node)


        dbg(f"bridge {bridge.pos}")
        dbg(f"f {f_path[-1].pos} s {self.start.pos}")
        dbg(f"b {b_path[-1].pos} g {self.goal.pos}")
        assert f_path[-1].pos == self.start.pos
        assert b_path[-1].pos == self.goal.pos
        
        if direction == START_END:
            f_path.reverse()
            b_path.pop(0)
            f_path.extend(b_path)
            return f_path
        elif direction == GOAL_END:
            b_path.reverse()
            f_path.pop(0)
            b_path.extend(f_path)
            return b_path
        else:
            raise Exception("Bad direction")

    def __get_node_path(self, bridge:Optional[Node]=None, direction:int=START_END) -> List[Node]:
        if bridge is None:
            bridge = self.main_bridge

        if bridge is None:
            return []

        return self.___get_node_path(bridge, direction)

    def get_path(self, bridge:Optional[Node]=None, direction:int=START_END) -> List[tuple]:
        node_path = self.__get_node_path(bridge, direction)
        path = [n.pos for n in node_path]

        return path

    def __update_total_cost(self) -> None:
        # Reset costs to all nodes
        for d in self.node_dict:
            for v in d.values():
                v.cost = float("inf")

        # Update costs to all connected nodes
        self.start.update_costs_rec()
        self.goal.update_costs_rec()

        # If no main_bridge, then we do not have interconnecting trees
        if self.main_bridge is None:
            self.total_cost = float("inf")
            return 

        f_end: Optional[Node] = self.forward_dict.get(self.main_bridge.pos)
        b_end: Optional[Node] = self.bacward_dict.get(self.main_bridge.pos)
        assert f_end is not None and b_end is not None

        self.total_cost = f_end.cost + b_end.cost
        dbg(f"New best cost: {self.total_cost}")

    def __optimize_bridge(self) -> None:
        for br in self.bridge_dict.values():
            f_node = self.forward_dict.get(br.pos)
            b_node = self.bacward_dict.get(br.pos)
            assert f_node is not None
            assert b_node is not None

            new_cost = f_node.cost + b_node.cost
            if new_cost < self.total_cost:
                dbg(f"New best bridge:{br.pos} cost:{new_cost}")
                self.total_cost = new_cost
                self.main_bridge = br

    def __plan_step(self) -> bool:
        end = START_END if len(self.forward_dict) < len(self.bacward_dict) else GOAL_END
        random_node = self.__get_random_node(end)
        nearest_node = self.__get_nearest_conn_node(random_node, end)
        new_node = self.__steer(nearest_node, random_node)

        if not self.__add_node(nearest_node, new_node, end):
            return False

        bridge = self.bridge_dict.get(new_node.pos)
        if bridge and self.main_bridge is not None and bridge.pos != self.main_bridge.pos:
            # We have new possible connection
            pass
            
        if self.main_bridge is not None:
            return True

        return False

    def plan(self, force_iters=None) -> bool:
        # Check for existing path
        if force_iters is None:
            node_path = self.__get_node_path()
            if len(node_path) > 0:
                return True

        for i in range(self.max_iter):
            status_ok  = self.__plan_step()
            # self.validate_nodes()
            if status_ok and force_iters is None:
                break
            elif force_iters is not None and i + 1 == force_iters:
                break


        self.__optimize_bridge()

        node_path = self.__get_node_path()
        return len(node_path) != 0

    def update_start(self, pos:tuple) -> bool:
        if pos == self.goal.pos:
            return False

        return self.__set_new_root(pos, START_END)

    def update_goal(self, pos:tuple) -> bool:
        if pos == self.start.pos:
            return False

        return self.__set_new_root(pos, GOAL_END)

    def validate_nodes(self) -> None:
        # Check parent-child relations for forward_dict
        for n in self.forward_dict.values():
            n.validate_relationships()

        # Check parent-child relations for bacward_dict
        for n in self.bacward_dict.values():
            n.validate_relationships()

        # Check if no nodes in both trees and bridge_dict
        # in obstacle positions
        for n in self.obst_dict.keys():
            assert self.forward_dict.get(n) is None
            assert self.bacward_dict.get(n) is None
            assert self.bridge_dict.get(n) is None

        # Check that all bridge_dict nodes has equivalent in
        # both trees
        for n in self.bridge_dict.keys():
            assert self.forward_dict.get(n) is not None
            assert self.bacward_dict.get(n) is not None

        # Check if nodes with same position that exists in 
        # both trees exists in bridge_dict
        for n in self.forward_dict.keys():
            n2 = self.bacward_dict.get(n)
            if n2 is not None:
                assert self.bridge_dict.get(n) is not None

        # Check roots
        for i in range(2):
            n_roots = 0
            for n in self.node_dict[i].values():
                assert (n.parent is not None and not n.is_root) or \
                        (n.parent is None and n.is_root)
                if n.parent is None and n.is_root:
                    n_roots += 1
            assert n_roots == 1

    def plot(self, plot_path:bool=True) -> None:
        import matplotlib.pyplot as plt
        f_nodes = self.forward_dict.values()
        b_nodes = self.bacward_dict.values()
        obstacle_list = self.obst_dict.keys()
        
        path = self.get_path()
        
        fx_paths = []
        fy_paths = []
        bx_paths = []
        by_paths = []

        for node in b_nodes:
            if node.parent is not None:
                bx_paths.append([node.parent.pos[0], node.pos[0]])
                by_paths.append([node.parent.pos[1], node.pos[1]])

        for node in f_nodes:
            if node.parent is not None:
                fx_paths.append([node.parent.pos[0], node.pos[0]])
                fy_paths.append([node.parent.pos[1], node.pos[1]])

        br_x = []
        br_y = []
        for node in self.bridge_dict.values():
            br_x.append(node.pos[0])
            br_y.append(node.pos[1])


        plt.figure()

        for (ox, oy) in obstacle_list:
            circle = plt.Circle((ox, oy), 0.3, color='r')
            plt.gca().add_patch(circle)


        for x_lst, y_lst in zip(fx_paths, fy_paths):
            plt.plot(x_lst, y_lst, color="blue")

        for x_lst, y_lst in zip(bx_paths, by_paths):
            plt.plot(x_lst, y_lst, color="red", linestyle='dashed')

        for node in f_nodes:
            plt.scatter([node.pos[0]], [node.pos[1]], color="blue", s=50)

        for node in b_nodes:
            plt.scatter([node.pos[0]], [node.pos[1]], color="red", s=60)

        if plot_path and path is not None:
            plt.plot([x for (x, _) in path], [y for (_, y) in path], color="green")
            # plt.scatter([x for (x, _) in path], [y for (_, y) in path], color="green")

        plt.scatter(br_x, br_y, color="orange", s=80)

        if self.main_bridge is not None:
            plt.scatter([self.main_bridge.pos[0]], [self.main_bridge.pos[1]], \
                    color="black", s=80)


        for node in f_nodes:
            plt.text(node.pos[0] - 0.5, node.pos[1] + 0.5,
                     f"{round(node.cost, 1)}",
                     fontsize=7)
        for node in b_nodes:
            plt.text(node.pos[0] - 0.0, node.pos[1] + 0.5,
                     f"{round(node.cost, 1)}",
                     fontsize=7)

        plt.plot(self.start.pos[0], self.start.pos[1], "xr")
        plt.plot(self.goal.pos[0], self.goal.pos[1], "xb")
        plt.xticks(list(range(self.shp[0]+1)))
        plt.yticks(list(range(self.shp[1]+1)))
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    start = (7, 1)
    goal = (17, 17)
    obstacle_list = [(10,9, -1), (6, 9, -1), (14, 10, -1), (12, 9, -1), (7, 8, -1), (15, 11, -1)]
    shp = (20, 20)

    m = DRRT(start, goal, shp, step_size=1, max_iter=200, goal_sample_rate=0.2)
    m.plan()
    m.validate_nodes()
    # m.plot()
    m.update_obstacles(obstacle_list)  # Update the tree with the new obstacles
    m.validate_nodes()
    m.plan()
    m.validate_nodes()
    # m.plot()

    obstacle_list.append((7, 10, -1))
    obstacle_list.append((12, 4, -1))
    obstacle_list.append((13, 9, -1))
    obstacle_list.append((8, 8, -1))
    obstacle_list.append((7, 9, -1))
    obstacle_list.append((8, 9, -1))
    obstacle_list.append((9, 9, -1))
    obstacle_list.append((11, 9, -1))
    m.update_obstacles(obstacle_list)
    # m.plot()
    m.plan()
    path = m.get_path()
    print(f"path={path}")
    # m.plot()

    obstacle_list.append((10, 13, -1))
    m.update_obstacles(obstacle_list)
    # m.plot()
    m.plan()

    for _ in range(50):
        m.plan(force_iters=20)
        m.validate_nodes()
        # m.plot()

    for i in range(3000):
        print(f"test iter:{i}")
        new_x = random.randint(0, m.shp[0])
        new_y = random.randint(0, m.shp[1])
        new_pos = (new_x, new_y)
        end_mod = random.randint(0,10)
        if end_mod % 2 == 0:
            dbg(f"Updating start to {new_pos}")
            status = m.update_start(new_pos)
            dbg(f"status={'OK' if status else 'FAIL'}")
        else:
            dbg(f"Updating goal to {new_pos}")
            status = m.update_goal(new_pos)
            dbg(f"status={'OK' if status else 'FAIL'}")

        m.validate_nodes()
        m.plan()

        list_length = random.randint(5, 100)
        random_tuples = [(random.randint(0, shp[0]-1), \
                random.randint(0, shp[1]-1), random.randint(-1,0)) for _ in range(list_length)]
        
        m.update_obstacles(random_tuples, clear=True)  # Update the tree with the new obstacles
        m.validate_nodes()
        m.plan(force_iters=random.randint(0, 200))
        # m.plot()
        m.validate_nodes()
        m.plan()
        m.validate_nodes()

        list_length = random.randint(5, 150)
        random_tuples = [(random.randint(0, shp[0]-1), \
                random.randint(0, shp[1]-1), random.randint(-1,0)) for _ in range(list_length)]

        m.update_obstacles(random_tuples, clear=False)  # Update the tree with the new obstacles
        m.plan()
        m.validate_nodes()
        
        
        
        

