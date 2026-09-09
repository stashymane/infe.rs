# Consumer R8/ProGuard rules for the UniFFI/JNA FFI layer.

# See: https://github.com/java-native-access/jna/blob/master/www/FrequentlyAskedQuestions.md#jna-on-android
# Gobley's generated rules omit "-keep class * extends com.sun.jna.*" (gobley#285).

# JNA
-dontwarn java.awt.**
-keep class com.sun.jna.** { *; }
-keep class * extends com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-keepclassmembers class * extends com.sun.jna.** { public *; }
-keepclassmembers class * extends com.sun.jna.Structure { <fields>; }
-keep class * implements com.sun.jna.Callback { *; }

# @Structure.FieldOrder and nested ByValue/ByReference types are resolved at runtime.
-keepattributes RuntimeVisibleAnnotations,RuntimeVisibleTypeAnnotations,AnnotationDefault,InnerClasses,EnclosingMethod,Signature

# UniFFI-generated bindings in this package (JNA maps them by name).
-keep class dev.stashy.infers.ffi.** { *; }
