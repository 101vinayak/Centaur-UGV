
class caliberation :
	def __init__(self , ret , mtx , dist , rvecs , tvecs,h,w,newcameramtx,roi ):
		self.ret = ret
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs
		self.mtx = mtx
		self.h = h
		self.w = w
		self.newcameramtx = newcameramtx
		self.roi = roi

	def _print_vals():
		print self.ret,' = ret'
		print self.dist,' = dist'
		print self.rvecs,' = rvecs'
		print self.tvecs,' = tvecs'
		print self.mtx,' = mtx'
		print self.h,' = h'
		print self.newcameramtx,' = newcameramtx'
		print self.roi,' = roi'