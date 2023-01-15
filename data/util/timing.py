import time


class TimeTester():
    def __init__(self,name,disabled=False):
        self.disabled=disabled
        if self.disabled: return
        self.start_time=time.time()
        self.start_dict={}
        self.stop_dict={}
        self.name=name
    def start(self,name):
        if self.disabled: return
        self.start_dict[name]=time.time()
        self.prev = name
    def end(self,name):
        if self.disabled: return
        self.stop_dict[name]=time.time()-self.start_dict[name]
    def end_prev(self):
        if self.disabled: return
        self.stop_dict[self.prev]=time.time()-self.start_dict[self.prev]
    def end_all(self):
        if self.disabled: return
        total = time.time()-self.start_time
        print(f"Time summary for {self.name}:")
        print(f"    Total time: {total}")
        for key in self.stop_dict.keys():
            print(f"    {key}: {self.stop_dict[key]/total*100:.2f}%")
        sub_total = sum(self.stop_dict.values())
        print(f"    Other: {(total-sub_total)/total*100:.2f}%")
