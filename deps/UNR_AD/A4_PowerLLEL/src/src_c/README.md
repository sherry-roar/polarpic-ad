# C Binding

尽量在组织结构上跟原本的src对齐，暂定操作方法：

1. 在该目录下新建同名文件，例如main.f90 -> main.c
2. 添加对应的C函数实现

```c
void c_binding_test() {
    printf("c_binding_test\n");
}
```

1. 在原有.f90文件内添加interface并调用

```f90
interface
    subroutine c_binding_test() bind(C)
        use, intrinsic :: iso_c_binding
    end subroutine
end interface
! ...
call c_binding_test()
```

如果整文件重写，将引用了该文件的use语句改为interface即可

改动比较多的话，这样搞Fortran和C互换应该也比较方便？

## TODO

1. Fortran数组的绑定传参
2. C的占比多了的话可能需要新增include目录
