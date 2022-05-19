from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from mytorch import torch_mnist

digit_reco = -1
# Create your views here.
def HTMLTemplate(articleTag, id=None):
    global digit_reco
    result= ''
    
    if digit_reco != -1:
        result =f'<h2>your answer is {digit_reco}</h2>'
    
    return f'''
    <html>
    <body>
        {articleTag}
        {result}
        
         <form action="/modeling/" enctype="multipart/form-data" method="post">
            <p>digit file upload</p>
            <input type="file" name="formFile" id="testfile">
            <input type="submit">
         </form>   
        

    </body>
    </html>                                 
    '''


def index(request):
    global digit_reco
    article='''
    <h2>welcome</h2>
    '''
    
    return HttpResponse(HTMLTemplate(article))



@csrf_exempt
def modeling(request):
    global digit_reco
    digit_reco = 3
    if request.method == 'POST':
        myfile = request.FILES['formFile']
        fs = FileSystemStorage()
        filename = fs.save('img/input.jpg',myfile)
        #print(filename)
        digit_reco= torch_mnist.upload_img_check(filename)
        #print(digit_reco)
  
    
    return redirect('/')