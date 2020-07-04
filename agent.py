class Agent():
    def __init__(self, x=None, y=None, bound_x=None, bound_y=None):
        self.x = x
        self.y = y
        self.bound_x = bound_x
        self.bound_y = bound_y
        self.state = None
        self.update_state()
        self.action_names = ['up', 'down', 'left', 'right']
        self.action_2_index = {el: ix for ix, el in enumerate(self.action_names)}

    def update_state(self):
        self.state = f'{self.y}|{self.x}'

    def check_boundaries(self,y,x):
        if 0 <= x < self.bound_x and 0 <= y < self.bound_y:
            return y, x
        else:
            if x < 0:
                x = 0
            elif x > self.bound_x:
                x = self.bound_x

            if y < 0:
                y = 0
            elif y > self.bound_y:
                y = self.bound_y
        return y, x

    def check_obstucles(self, y,x, cur_y, cur_x, obstucles):
        if obstucles is None:
            return y, x

        for lu, ru, ld, rd in obstucles:
            if lu[1] <= x <= rd[1] and lu[0] <= y <= rd[0]:
                x = cur_x
                y = cur_y
        return y, x

    def action(self, action, obstacles):
        tmp_y = self.y
        tmp_x = self.x
        if action == 'up':
            tmp_y = self.y - 1
        elif action == 'down':
            tmp_y = self.y + 1
        elif action == 'left':
            tmp_x = self.x - 1
        elif action == 'right':
            tmp_x = self.x + 1

        tmp_y, tmp_x = self.check_boundaries(tmp_y, tmp_x, )
        tmp_y, tmp_x = self.check_obstucles(tmp_y, tmp_x, self.y, self.x, obstacles)

        self.y = tmp_y
        self.x = tmp_x

        self.update_state()

    def get_actions_names(self):
        return self.action_names

    def get_state(self):
        return self.state
